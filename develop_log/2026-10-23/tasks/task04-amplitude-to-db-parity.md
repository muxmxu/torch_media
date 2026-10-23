# Task 04 — amplitude_to_DB parity (top_db packing, defaults, precision)
id: 2026-10-23/task04
parent: 2026-10-23/progress02
status: done
owner: code_agent

## Objective
`functional::amplitude_to_DB` and `transform::AmplitudeToDB` match torchaudio 2.5.1 for any rank and dtype.

## Findings
G1-1, G3-1 (top_db cutoff per [F, T] slice), G1-2 (default db_multiplier 1 / top_db on), G1-6 (FLT_MIN amin
floor), G1-7 (float parameters), new: AmplitudeToDB accepts a negative top_db.

## Scope
In: `_functional.hpp` (amplitude_to_DB, db_to_amplitude, mfcc's call), `_functional_methods_options.hpp`
(amplitude_to_db_option, mfcc_option::top_db), `_transform_feature.hpp` (AmplitudeToDB).
Out: complex input (G1-3, still abs(); see progress02 Open).

## Deliverables
- Cutoff over (-3, -2, -1) of the input packed as (-1, C, F, T); 1-D + top_db throws.
- Option defaults = torchaudio AmplitudeToDB(); `set_top_db()` enables the cutoff; double fields.
- `AmplitudeToDB` throws for top_db < 0; passes amin 1e-10 / top_db as double.
- Tests: `test_amplitude_to_DB_top_db_packing`, `test_amplitude_to_DB_defaults_amin_precision` (functional),
  `test_amplitude_to_db_parity` (transform); goldens in functional gen_golden.py.

## Acceptance criteria
- [x] Red on e7041b5: 3-D/4-D cutoff off by 19.65 dB, default off by 10 dB, amin=0 not -inf (|delta| 358.5),
  float64 params off (1e-7), 1-D top_db raised c10::Error not std::invalid_argument, negative top_db accepted,
  transform float64 off by 5.8e-8.
- [x] Green after the fix; probe on random 3/4/5-D inputs (top_db 40/80, multiplier 10/20, defaults) bit-exact.
- [x] ctest green.
