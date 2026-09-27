import numpy as np

from resemblyzer.audio import float_to_int16


def test_out_of_range_samples_do_not_wrap():
    pcm = float_to_int16(np.array([1.5, -1.5, 1.0, -1.0, 0.0]))
    assert list(int(x) for x in pcm) == [32767, -32767, 32767, -32767, 0]
