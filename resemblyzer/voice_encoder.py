from resemblyzer.hparams import *
from resemblyzer import audio
from pathlib import Path
from typing import Union
from torch import nn
import numpy as np
import torch


class VoiceEncoder(nn.Module):
    def __init__(self, device: Union[str, torch.device]=None):
        """
        :param device: either a torch device or the name of a torch device (e.g. "cpu", "cuda"). 
        If None, defaults to cuda if it is available on your machine, otherwise the model will 
        run on cpu. Outputs are always returned on the cpu, as numpy arrays.
        """
        super().__init__()
        
        # Define the network
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()
        
        # Get the target device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
            
        # Load the pretrained model's weights
        weights_fpath = Path(__file__).resolve().parent.joinpath("pretrained.pt")
        if not weights_fpath.exists():
            raise Exception("Couldn't find the voice encoder pretrained model at %s." % 
                            weights_fpath)
        checkpoint = torch.load(weights_fpath, map_location="cpu")
        self.load_state_dict(checkpoint["model_state"], strict=False)
        self.to(device)

    def forward(self, mels: torch.FloatTensor):
        """
        Computes the embeddings of a batch of utterance spectrograms.

        :param mels: a batch of mel spectrograms of same duration as a float32 tensor of shape 
        (batch_size, n_frames, n_channels) 
        :return: the embeddings as a float 32 tensor of shape (batch_size, embedding_size). 
        Embeddings are positive and L2-normed, thus they lay in the range [0, 1].
        """
        # Pass the input through the LSTM layers and retrieve the final hidden state of the last 
        # layer. Apply a cutoff to 0 for negative values and L2 normalize the embeddings.
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)
    
    @staticmethod
    def compute_partial_slices(n_samples: int, min_coverage=0.75, overlap=0.5):
        """
        Computes where to split an utterance waveform and its corresponding mel spectrogram to 
        obtain partial utterances of <partials_n_frames> each. Both the waveform and the 
        mel spectrogram slices are returned, so as to make each partial utterance waveform 
        correspond to its spectrogram.
    
        The returned ranges may be indexing further than the length of the waveform. It is 
        recommended that you pad the waveform with zeros up to wave_slices[-1].stop.
    
        :param n_samples: the number of samples in the waveform
        :param min_coverage: when reaching the last partial utterance, it may or may not have 
        enough frames. If at least <min_pad_coverage> of <partials_n_frames> are present, 
        then the last partial utterance will be considered by zero-padding the audio. Otherwise, 
        it will be discarded. If there aren't enough frames for one partial utterance, 
        this parameter is ignored so that the function always returns at least one slice.
        :param overlap: by how much the partial utterance should overlap. If set to 0, the partial 
        utterances are entirely disjoint. 
        :return: the waveform slices and mel spectrogram slices as lists of array slices. Index 
        respectively the waveform and the mel spectrogram with these slices to obtain the partial 
        utterances.
        """
        assert 0 <= overlap < 1
        assert 0 < min_coverage <= 1
        
        samples_per_frame = int((sampling_rate * mel_window_step / 1000))
        n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
        frame_step = max(int(np.round(partials_n_frames * (1 - overlap))), 1)
        
        # Compute the slices
        wav_slices, mel_slices = [], []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            mel_range = np.array([i, i + partials_n_frames])
            wav_range = mel_range * samples_per_frame
            mel_slices.append(slice(*mel_range))
            wav_slices.append(slice(*wav_range))
        
        # Evaluate whether extra padding is warranted or not
        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(mel_slices) > 1:
            mel_slices = mel_slices[:-1]
            wav_slices = wav_slices[:-1]
        
        return wav_slices, mel_slices
    
    def embed_utterance(self, wav: np.ndarray, return_partials=False, **kwargs):
        """
        Computes an embedding for a single utterance.
    
        :param wav: a preprocessed utterance waveform as a numpy array of float32
        :param return_partials: if True, the partial embeddings will also be returned along with the 
        wav slices that correspond to the partial embeddings.
        :param kwargs: additional arguments to compute_partial_splits()
        :return: the embedding as a numpy array of float32 of shape (model_embedding_size,). If 
        <return_partials> is True, the partial utterances as a numpy array of float32 of shape 
        (n_partials, model_embedding_size) and the wav partials as a list of slices will also be 
        returned. If <using_partials> is simultaneously set to False, both these values will be None 
        instead.
        """
        # Compute where to split the utterance into partials and pad if necessary
        wave_slices, mel_slices = self.compute_partial_slices(len(wav), **kwargs)
        max_wave_length = wave_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
        
        # Split the utterance into partials
        mel = audio.wav_to_mel_spectrogram(wav)
        mels = np.array([mel[s] for s in mel_slices])
        with torch.no_grad():
            mels = torch.from_numpy(mels).to(self.device)
            partial_embeds = self(mels).cpu().numpy()
        
        # Compute the utterance embedding from the partial embeddings
        raw_embed = np.mean(partial_embeds, axis=0)
        embed = raw_embed / np.linalg.norm(raw_embed, 2)
        
        if return_partials:
            return embed, partial_embeds, wave_slices
        return embed
    