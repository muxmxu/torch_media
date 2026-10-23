# Task 32 — readme: metric and data
id: 2026-10-24/task32
parent: 2026-10-24/progress04
status: done
owner: code_agent

## Objective
The readme describes `torchmedia::metric` and `torchmedia::data` (what each provides, its parity reference) with one
metric and one data example, in the existing style.

## Scope
In:
- `readme.md`: table rows (vision::io updated; metric detection / image + segmentation / audio / text /
  classification; data), "Metric example", "Data example".
Out:
- Rewriting other sections; the hobby / no-liability notice stays as is.

## Acceptance criteria
- [x] Every snippet is copied from a program compiled and run against the worktree headers through the `torch_media`
  target (scratch `pnm/readme-check`), and the stated outputs are that program's output.
- [x] The metric output equals torchmetrics 1.9 (pycocotools backend) on the same input.
- [x] Inline code fragments in the prose compile (`inline.cpp`).
