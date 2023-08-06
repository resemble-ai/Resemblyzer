from resemblyzer import preprocess_wav, VoiceEncoder
from demo_utils import *
from itertools import groupby
from pathlib import Path
from tqdm import tqdm
import numpy as np


# DEMO 03: we'll show one way to visualize these utterance embeddings. Since they are 
# 256-dimensional, it is much simpler for us to get an overview of their manifold if we reduce 
# their dimensionality first. By doing so, we can observe clusters that form for utterances of 
# identical characteristics. What we'll see is that clusters form for distinct speakers, 
# and they are very tight and even linearly separable.


## Gather the wavs
wav_fpaths = list(Path("audio_data", "librispeech_test-other").glob("**/*.flac"))
speakers = list(map(lambda wav_fpath: wav_fpath.parent.stem, wav_fpaths))
wavs = np.array(list(map(preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths)))), dtype=object)
speaker_wavs = {speaker: wavs[list(indices)] for speaker, indices in 
                groupby(range(len(wavs)), lambda i: speakers[i])}


## Compute the embeddings
encoder = VoiceEncoder()
utterance_embeds = np.array(list(map(encoder.embed_utterance, wavs)))


## Project the embeddings in 2D space
plot_projections(utterance_embeds, speakers, title="Embedding projections")
plt.show()
