# Task 21 — Data: ManifestDataset, PadCollate, Subset / ConcatDataset / random_split
id: 2026-10-24/task21
parent: 2026-10-24/progress03
status: done
owner: code_agent

## Objective
Add a manifest-file dataset, a padding collation for variable-length samples, and torch.utils.data's Subset,
ConcatDataset and random_split (torch 2.5.1 semantics, seeded parity).

## Scope
In:
- `_data/_manifest.hpp` (ManifestDataset, manifest_option), `_data/_collate.hpp` (PadCollate, padded_batch,
  pad_collate_option), `_data/_dataset.hpp` (Subset, ConcatDataset, random_split, `detail::py_sum`).
Out:
- Quoted TSV fields, Kaldi pipes / ark offsets; ConcatDataset over different dataset types.

## Inputs (read first, priority order)
1. `torch/utils/data/dataset.py` (2.5.1) — Subset, ConcatDataset, random_split.
2. `torch/data/transforms/{base,collate,stack}.h` — BatchTransform / Collation / Stack.
3. CPython 3.12 `builtin_sum_impl` — the compensated float sum random_split relies on.

## Deliverables
- The API of progress03 D4-D6.

## Acceptance criteria
- [x] random_split equals torch's split for the same seed (explicit generator and global generator) incl. fraction
  rounding and Python 3.12's sum (task22).
- [x] PadCollate usable as `dataset.map(PadCollate())` with the loader; lengths, pad value, pad-to-multiple, validation.
- [x] Manifest errors name the manifest and line.

## Constraints
- As task20.
