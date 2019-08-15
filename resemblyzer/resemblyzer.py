from resemblyzer.encoder.inference import embed_utterance
import numpy as np


def similarity(embed: np.ndarray, wav: np.ndarray, rate=8):
    overlap = 1 - 1 / rate
    _, partial_embeds, wav_slices = embed_utterance(wav, return_partials=True, overlap=overlap)
    similarities = partial_embeds @ embed
    return similarities, wav_slices
    


