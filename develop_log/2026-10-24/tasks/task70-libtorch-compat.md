# Task 70 — Build with any compatible libtorch
id: 2026-10-24/task70
parent: 2026-10-24/progress08
status: done
owner: code_agent

## Objective
Remove the headers' use of the C++ frontend `torch::linalg::` namespace and document what another libtorch version
changes.

## Acceptance criteria
- [x] `torchmedia.hpp` compiles against libtorch 2.5.1 and 2.14.1.
- [x] ctest 24/24 on 2.5.1 (results unchanged: same kernels).
- [x] The 2.14.1 test run is recorded (pass / fail counts and the size of every difference class) in progress08 and
  doc/reproducibility.md.
