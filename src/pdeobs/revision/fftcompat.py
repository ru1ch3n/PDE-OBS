"""cuFFT-compatible inverse real 2-D FFT for CPU replays of GPU-trained FNO checkpoints.

The FNO spectral layer (`pdeobs.methods.neural.SpectralConv2d`) has independent weights for +ky and -ky. Its truncated
half spectrum is therefore not conjugate-symmetric in the kx = 0 column (nor, in general, in the Nyquist column). The real
inverse transform then has to discard information, and libraries discard it differently.

All of them evaluate `irfft2` as an inverse complex FFT along y followed by a 1-D complex-to-real (C2R) transform along x.
In that C2R, the imaginary parts of the kx = 0 and kx = n/2 bins have no real counterpart:
- CPU FFTs (MKL, pocketfft) drop them.
- cuFFT drops them for small batches.
- For large batches (observed from 4,096 rows of 1-D transforms; 1,920 rows still drop), cuFFT uses a kernel that adds
  -(-1)^j * Im X[0] / sqrt(n) and -Im X[n/2] / sqrt(n) to output sample j (norm "ortho").

Every October FNO was trained and evaluated on NVIDIA GPUs with batch x channels x H far above that size, so its
weights expect the large-batch cuFFT result. On a CPU the same checkpoint gives different predictions: up to 1.2e-2
relative change of a view's mean error was measured on Poisson U256 / G256. `irfft2_cufft` reproduces the large-batch
cuFFT result on any device (agreement with cuFFT ~1e-7 absolute in float32).

Measured 2026-10-16: PyTorch 2.14.1, cuFFT from CUDA 13.0, NVIDIA RTX PRO 6000 Blackwell; CPU MKL. The batch threshold
is a property of cuFFT's kernel selection and may change with the library version or the GPU.
"""
from __future__ import annotations

import contextlib
import math

import torch

_NATIVE_IRFFT2 = torch.fft.irfft2


def irfft2_cufft(input, s=None, dim=(-2, -1), norm=None, *, out=None):
    """`torch.fft.irfft2` over the last two dimensions with cuFFT's large-batch C2R treatment of Im X[..., 0] and
    Im X[..., n/2] (norm "ortho", even output width). Other calls are refused rather than approximated."""
    if out is not None or tuple(dim) not in ((-2, -1), (input.ndim - 2, input.ndim - 1)) or norm != "ortho":
        raise NotImplementedError("irfft2_cufft covers dim=(-2, -1), norm='ortho' and out=None (the FNO layer's call)")
    h = int(input.shape[-2] if s is None else s[0])
    w = int(2 * (input.shape[-1] - 1) if s is None else s[1])
    if w % 2:
        raise NotImplementedError("irfft2_cufft needs an even output width")
    y = torch.fft.ifft(input, n=h, dim=-2, norm="ortho")
    result = torch.fft.irfft(y, n=w, dim=-1, norm="ortho")
    sign = 1.0 - 2.0 * (torch.arange(w, device=result.device) % 2).to(result.dtype)       # (-1)^j
    leak = y[..., 0].imag.to(result.dtype)[..., None] * sign
    if y.shape[-1] > w // 2:
        leak = leak + y[..., w // 2].imag.to(result.dtype)[..., None]
    return result - leak / math.sqrt(w)


@contextlib.contextmanager
def cufft_c2r_on_cpu():
    """Within the block, `torch.fft.irfft2` on CPU tensors follows `irfft2_cufft`; CUDA tensors keep the native call."""
    def patched(input, s=None, dim=(-2, -1), norm=None, *, out=None):
        if input.device.type == "cpu":
            return irfft2_cufft(input, s=s, dim=dim, norm=norm, out=out)
        return _NATIVE_IRFFT2(input, s=s, dim=dim, norm=norm, out=out)

    previous = torch.fft.irfft2
    torch.fft.irfft2 = patched
    try:
        yield
    finally:
        torch.fft.irfft2 = previous


__all__ = ["cufft_c2r_on_cpu", "irfft2_cufft"]
