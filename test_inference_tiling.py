from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'studio-multiscroll-cpt-20261023/hf'))
import inference  # noqa: E402


class Stub(torch.nn.Module):
    """Stands in for the U-Net: a per-pixel function of the 16 planes, counting the windows it sees."""
    def __init__(self):
        super().__init__(); self.windows = 0

    def forward(self, x):
        self.windows += x.shape[0]
        return x.mean(2) * 8 - 2 + x.std(2)


def surface(h=500, w=700, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.integers(1, 255, (16, h, w), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    v[:, (yy - h / 2) ** 2 / (h / 3) ** 2 + (xx - w / 2) ** 2 / (w / 3) ** 2 > 1] = 0   # no CT outside an ellipse
    return v


class SkipEmptyTests(unittest.TestCase):
    def run_both(self, stride, blend):
        v = surface(); out = {}
        with tempfile.TemporaryDirectory() as d:
            for skip in (False, True):
                m = Stub()
                p = inference.predict(m, v, Path(d) / f'{skip}.npy', device='cpu', batch_size=7, amp='fp32',
                                      stride=stride, blend=blend, skip_empty=skip)
                out[skip] = (np.array(p), m.windows)
        return out

    def test_same_output_fewer_windows(self):
        for stride, blend in ((96, 'none'), (64, 'hann'), (48, 'hann')):
            with self.subTest(stride=stride, blend=blend):
                out = self.run_both(stride, blend)
                np.testing.assert_array_equal(out[True][0], out[False][0])
                self.assertLess(out[True][1], out[False][1])

    def test_nothing_skipped_without_empty_pixels(self):
        v = np.full((16, 200, 300), 90, np.uint8)
        with tempfile.TemporaryDirectory() as d:
            m = Stub(); inference.predict(m, v, Path(d) / 'a.npy', device='cpu', amp='fp32', skip_empty=True)
        self.assertEqual(m.windows, 3 * 4)


if __name__ == '__main__':
    unittest.main()
