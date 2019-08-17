from resemblyzer import preprocess_wav, VoiceEncoder
from itertools import groupby
from pathlib import Path
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
wav_fpaths = list(Path("librispeech_test-other").glob("**/*.flac"))
# Group the wavs per speaker and load them using the preprocessing function provided with 
# resemblyzer to load wavs in memory. It normalizes the volume, trims long silences and resamples 
# the wav to the correct sampling rate.
spearker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in 
                 groupby(wav_fpaths, lambda wav_fpath: wav_fpath.parent.stem)} 


def demo_similarity_matrix():
    
    ## Similarity between two utterances from each speaker
    # Embed two utterances A and B for each speaker
    embeds_a = np.array([encoder.embed_utterance(wavs[0]) for wavs in spearker_wavs.values()])
    embeds_b = np.array([encoder.embed_utterance(wavs[-1]) for wavs in spearker_wavs.values()])
    # Each array is of shape (num_speakers, embed_size) which should be (10, 256) if you haven't 
    # changed anything.
    print("Shape of embeddings: %s" % str(embeds_a.shape))
    
    # Compute the similarity matrix. The similarity of two embeddings is simply their dot 
    # product, because the similarity metric is the cosine similarity and the embeddings are 
    # already L2-normed.
    # Short version:
    sim_matrix = np.inner(embeds_a, embeds_b)
    # Long, detailed version:
    sim_matrix2 = np.zeros((len(embeds_a), len(embeds_b)))
    for i in range(len(embeds_a)):
        for j in range(len(embeds_b)):
            # The @ notation is exactly equivalent to np.dot(embeds_a[i], embeds_b[i])
            sim_matrix2[i, j] = embeds_a[i] @ embeds_b[j]
    assert np.allclose(sim_matrix, sim_matrix2)
    
    
    ## Similarity between speaker embeddings and utterances 
    # Embed the speakers, excluding the utterance that will be used for comparison to avoid bias.  
    speaker_embeds = np.array([encoder.embed_speaker(wavs[1:]) for wavs in spearker_wavs.values()])
    sim_matrix = np.inner(speaker_embeds, embeds_a)
    breakpoint()
    
    
    
    


if __name__ == '__main__':
    demo_similarity_matrix()
    quit()

    # speaker_dirs = list(Path(r"E:\Datasets\LibriSpeech\train-clean-100").glob("*"))
    # n_speakers = 100
    # n_embeds = 15
    # 
    # wav_fpaths = list(speaker_dirs[0].glob("**/*.flac"))
    # target = preprocess_wav(wav_fpaths[0])
    # 
    # _, target_partials, wav_slices = encoder.embed_utterance(target, True)
    # 
    # for wav_fpath in wav_fpaths[:n_embeds]:
    #     wav = preprocess_wav(wav_fpath)
    #     embed = encoder.embed_utterance(wav)
    #     similarities = target_partials @ embed
    #     
    #     times = [(s.start + s.stop) // 2 for s in wav_slices]
    #     plt.plot(times, similarities)
    # 
    # span = range(0, len(target), 16000)
    # plt.xticks(span, map(lambda t: "%ds" % (t // 16000), span))
    # plt.ylim(0, 1)
    # plt.show()
    
    
    # wav_lens = []
    # for i, speaker_dir in enumerate(speaker_dirs[:n_speakers]):
    #     wav = audio.preprocess_wav(next(speaker_dir.glob("**/*.flac")))
    #     wav_lens.append(len(wav))
    #     similarities, wav_splits = similarity(embed, wav)
    #     times = [(s.start + s.stop) // 2 for s in wav_splits]
    #     plt.plot(times, similarities, label="speaker %d" % i)
    # span = range(0, max(wav_lens), 16000)
    # plt.legend()
    # plt.xticks(span, map(lambda t: "%ds" % (t // 16000), span))
    # plt.ylim(0, 1)
    # plt.show()
    # 