# Task 03 — Text metrics
id: 2026-10-24/task03
parent: 2026-10-24/progress01
status: done
owner: code_agent

## Objective
`torchmedia::metric::text`: torchmetrics 1.9 WER, CER, MER, WIL, WIP and EditDistance over UTF-8 strings, equal to
torchmetrics for any script.

## Scope
In:
- `_metric/_text.hpp`: UTF-8 decoding with validation, Python `str.split()` whitespace, float32 sums in torchmetrics'
  order, sacrebleu's beam-limited edit distance with torchmetrics' reductions (progress01 D6).
- Reuse `torchmedia::audio::functional::edit_distance` for the per-pair Levenshtein distance.
Out:
- BLEU / chrF / ROUGE / TER / EED / SQuAD / perplexity / model-based text metrics; jiwer's transforms.

## Acceptance criteria
- [x] Chinese / Japanese / mixed-script / emoji / empty strings and every separator kind bit-exact against
  torchmetrics; jiwer agrees on its single-space subset (asserted in gen_golden.py).
- [x] Length mismatch, invalid UTF-8 and negative substitution cost are `std::invalid_argument`.
