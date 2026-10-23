# Task 01 — torchmetrics audio metrics and STOI
id: 2026-10-24/task01
parent: 2026-10-24/progress01
status: done
owner: code_agent

## Objective
`torchmedia::metric::audio` functionals and classes equal to torchmetrics 1.9 SNR, SI-SDR, SI-SNR, C-SI-SNR, SDR,
SA-SDR and PIT, and STOI equal to pystoi 0.4.1.

## Scope
In:
- `_metric/_audio.hpp`: the functionals / classes of progress01 D1-D3 with their option structs, fast_bss_eval's
  Toeplitz conjugate gradient, scipy's `linear_sum_assignment` (square), pystoi's resampling / framing /
  silent-frame removal / 1/3-octave bands / clipping / extended normalisation.
- `_metric/_common.hpp`: `metric_base` default-constructs classes without options; `metric.hpp` includes the header.
Out:
- PESQ, DNSMOS, NISQA, SRMR, torchmetrics' `keep_same_device`.

## Inputs (read first, priority order)
1. `develop_log/2026-10-23/progress11-metric-vision.md` — layout, conventions, deliberate-difference policy.
2. torchmetrics `functional/audio/{snr,sdr,pit,stoi}.py`, `audio/{snr,sdr,pit,stoi}.py`; fast_bss_eval
   `torch/cgd.py`; pystoi `stoi.py`, `utils.py`; scipy `resample_poly`.

## Acceptance criteria
- [x] Same libtorch calls and order as torchmetrics (bit-exact goldens on arm64); STOI within 1e-12 of pystoi.
- [x] Option / input validation as progress01 D7 / D8; no-exceptions build compiles.
