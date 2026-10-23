#!/usr/bin/env python3
"""Golden reference values for the separation unit tests, computed with torchiva 0.1.1 (torch 2.5.1).

Run in the project venv:
    .venv/bin/python unit_test/audio/separation/gen_golden.py

Inputs are seeded `torch.randn` draws; libtorch's CPU generator produces the same numbers for the same seed,
and the NMF model draws its init (`uniform_`) from the same generator right after, so main.cpp rebuilds every
input bit-for-bit and compares against the constants printed here. Each case prints, for the output Y:
  |Y|^2 summed, the complex checksum sum(Y * conj(Z)) with Z = randn(Y.shape, complex128) under seed 99,
  and four elements of the last batch item (s, f, n, re, im).
"""
import torch
import torchiva
from torchiva.models import GaussModel, LaplaceModel, NMFModel


def report(name, Y):
    Yd = Y.to(torch.complex128)
    s, f, n = Y.shape[-3:]
    torch.manual_seed(99)
    chk = (Yd * torch.randn(Y.shape, dtype=torch.complex128).conj()).sum()
    last = Yd.reshape(-1, s, f, n)[-1]
    idx = [(0, 0, 0), (0, f // 2, n // 2), (s - 1, f // 2, n // 3), (s - 1, f - 1, n - 1)]
    els = ", ".join(f"{{{a}, {b}, {c}, {last[a, b, c].real.item():.12g}, {last[a, b, c].imag.item():.12g}}}"
                    for a, b, c in idx)
    print(f"// {name}: shape {tuple(Y.shape)}")
    print(f"    {{{Yd.abs().square().sum().item():.12g}, {chk.real.item():.12g}, {chk.imag.item():.12g}, {{{els}}}}}")


def run(seed, shape, dtype, model="laplace", n_src=None, mic=0, n_iter=10, n_basis=2, eps=1e-5):
    torch.manual_seed(seed)
    X = torch.randn(*shape, dtype=dtype)
    # torchiva's separator and model eps e map to C++ eps(e).load(e); with eps != 1e-5 use mic=None, because torchiva's
    # projection back ignores its load (fixed 1e-5) while C++ uses auxiva_option::load there
    model = {"laplace": lambda: LaplaceModel(eps=eps), "gauss": lambda: GaussModel(eps=eps),
             "nmf": lambda: NMFModel(n_basis=n_basis, eps=eps)}[model]()
    n_src = shape[-3] if n_src is None else n_src  # torchiva's n_src=None path crashes; C++ uses -1 => n_chan
    return torchiva.AuxIVA_IP(n_iter=n_iter, n_src=n_src, model=model, proj_back_mic=mic, eps=eps)(X)


c64, c128 = torch.complex64, torch.complex128
report("laplace", run(0, (3, 17, 50), c64))
report("gauss", run(0, (3, 17, 50), c64, model="gauss"))
report("no proj back", run(0, (3, 17, 50), c64, mic=None))
report("mic1", run(0, (3, 17, 50), c64, mic=1))
report("n_iter 0", run(0, (3, 17, 50), c64, n_iter=0))
report("eps 1e-3 gauss no proj back", run(0, (3, 17, 50), c64, model="gauss", mic=None, eps=1e-3))
report("nmf", run(1, (2, 17, 50), c64, model="nmf"))
report("complex128 nmf rank3", run(4, (2, 17, 50), c128, model="nmf", n_basis=3))
report("batch nmf", run(7, (2, 2, 9, 30), c64, model="nmf"))
report("overiva mic0", run(2, (3, 17, 50), c64, n_src=2))
# torchiva re-broadcasts the OverIVA projection for mic >= n_src (output [..., F, ..., S, F, N]). Its leading slices
# equal the per-frequency result only up to rounding (bit-exact at this small F; at F=257 they spread by up to 2.7e-4
# in complex64), so keep these cases small: [0] (or [0, 0] with a batch dim) is the reference.
report("overiva mic2", run(2, (3, 17, 50), c64, n_src=2, mic=2)[0])
report("batch overiva mic2", run(6, (2, 3, 17, 50), c64, n_src=2, mic=2)[0, 0])
report("overiva eps 1e-3 no proj back", run(2, (3, 17, 50), c64, n_src=2, mic=None, eps=1e-3))
report("overiva nmf n_src1 no proj back", run(8, (3, 17, 50), c64, model="nmf", n_src=1, mic=None))
report("complex128 laplace", run(3, (2, 17, 50), c128))
report("batch laplace", run(5, (2, 2, 17, 50), c64))
