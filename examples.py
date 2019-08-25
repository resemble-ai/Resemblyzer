from examples_utils import *
from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


# The neural network will automatically use CUDA if it's available on your machine, otherwise it 
# will use the CPU. You can enforce a device of your choice by passing its name as argument to the 
# constructor. The model might take a few seconds to load with CUDA, but it then executes very 
# quickly.
encoder = VoiceEncoder()

# We'll use a smaller version of the dataset LibriSpeech test-other to run our examples. This 
# smaller dataset contains 10 speakers with 10 utterances each. N.B. "wav" in variable names stands
# for "waveform" and not the wav file extension.
wav_fpaths = list(Path("audio_data", "librispeech_test-other").glob("**/*.flac"))
# Group the wavs per speaker and load them using the preprocessing function provided with 
# resemblyzer to load wavs in memory. It normalizes the volume, trims long silences and resamples 
# the wav to the correct sampling rate.
speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"), 
                        lambda wav_fpath: wav_fpath.parent.stem)}


def demo_similarity_matrix():
    ## Similarity between two utterances from each speaker
    # Embed two utterances A and B for each speaker
    embeds_a = np.array([encoder.embed_utterance(wavs[0]) for wavs in speaker_wavs.values()])
    embeds_b = np.array([encoder.embed_utterance(wavs[1]) for wavs in speaker_wavs.values()])
    # Each array is of shape (num_speakers, embed_size) which should be (10, 256) if you haven't 
    # changed anything.
    print("Shape of embeddings: %s" % str(embeds_a.shape))
    
    # Compute the similarity matrix. The similarity of two embeddings is simply their dot 
    # product, because the similarity metric is the cosine similarity and the embeddings are 
    # already L2-normed.
    # Short version:
    utt_sim_matrix = np.inner(embeds_a, embeds_b)
    # Long, detailed version:
    utt_sim_matrix2 = np.zeros((len(embeds_a), len(embeds_b)))
    for i in range(len(embeds_a)):
        for j in range(len(embeds_b)):
            # The @ notation is exactly equivalent to np.dot(embeds_a[i], embeds_b[i])
            utt_sim_matrix2[i, j] = embeds_a[i] @ embeds_b[j]
    assert np.allclose(utt_sim_matrix, utt_sim_matrix2)
    
    
    ## Similarity between two speaker embeddings
    # Embed the speakers, excluding the utterance that will be used for comparison to avoid bias.  
    spk_embeds_a = np.array([encoder.embed_speaker(wavs[:len(wavs) // 2]) \
                             for wavs in speaker_wavs.values()])
    spk_embeds_b = np.array([encoder.embed_speaker(wavs[len(wavs) // 2:]) \
                             for wavs in speaker_wavs.values()])
    spk_sim_matrix = np.inner(spk_embeds_a, spk_embeds_b)

    
    ## Draw the plots
    fix, axs = plt.subplots(2, 2, figsize=(8, 10))
    labels_a = ["%s-A" % i for i in speaker_wavs.keys()]
    labels_b = ["%s-B" % i for i in speaker_wavs.keys()]
    mask = np.eye(len(utt_sim_matrix), dtype=np.bool)
    plot_similarity_matrix(utt_sim_matrix, labels_a, labels_b, axs[0, 0],
                           "Cross-similarity between utterances\n(speaker_id-utterance_group)")
    plot_histograms((utt_sim_matrix[mask], utt_sim_matrix[np.logical_not(mask)]), axs[0, 1],
                    ["Same speaker", "Different speakers"], 
                    "Normalized histogram of similarity\nvalues between utterances")
    plot_similarity_matrix(spk_sim_matrix, labels_a, labels_b, axs[1, 0],
                           "Cross-similarity between speakers\n(speaker_id-utterances_group)")
    plot_histograms((spk_sim_matrix[mask], spk_sim_matrix[np.logical_not(mask)]), axs[1, 1],
                    ["Same speaker", "Different speakers"], 
                    "Normalized histogram of similarity\nvalues between speakers")
    plt.show()


def demo_speaker_diarization():
    ## Get reference audios
    # Load the interview audio from disk
    # Source for the interview: https://www.youtube.com/watch?v=X2zqiX6yL3I
    wav_fpath = Path("audio_data", "X2zqiX6yL3I.mp3")
    wav = preprocess_wav(wav_fpath)
    
    # Cut some segments from single speakers as reference audio
    segments = [[0, 5.5], [6.5, 12], [17, 25]]
    speaker_names = ["Kyle Gass", "Sean Evans", "Jack Black"]
    speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1]) * sampling_rate] for s in segments]
      
        
    ## Compare speaker embeds to the continuous embedding of the interview
    # Derive a continuous embedding of the interview. We put a rate of 16, meaning that an 
    # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker 
    # diarization, but it is not so useful for when you only need a summary embedding of the 
    # entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the 
    # demonstration. 
    # We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs 
    # won't have enough. There's a speed drawback, but it remains reasonable.
    cpu_encoder = VoiceEncoder("cpu")
    print("Running the continuous embedding on the CPU, this might take a while...")
    _, cont_embeds, wav_splits = cpu_encoder.embed_utterance(wav, return_partials=True, rate=16)
    
    # Get the continuous similarity for every speaker. It amounts to a dot product between the 
    # embedding of the speaker and the continuous embedding of the interview
    speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
    similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                       zip(speaker_names, speaker_embeds)}
    
    
    ## Run the interactive demo
    interactive_diarization(similarity_dict, wav, wav_splits)


if __name__ == '__main__':
    # demo_similarity_matrix()
    demo_speaker_diarization()
