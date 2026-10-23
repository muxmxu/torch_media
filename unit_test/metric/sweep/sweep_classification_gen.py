"""Randomised classification parity sweep (dev-time, not in ctest; develop_log/2026-10-24/progress02): cases from
gen_golden_classification.make_case, torchmetrics 1.9.0 values from its expected() (functional on the whole input,
class over two halves, "<key>!" where torchmetrics raised). From a scratch directory:
    .venv/bin/python <repo>/unit_test/metric/sweep/sweep_classification_gen.py 2000 1   # cases.pt, expected.pt
    <build>/unit_test/metric/metric_sweep_classification                                # -> results.pt
    .venv/bin/python <repo>/unit_test/metric/sweep/compare.py
Usage: sweep_classification_gen.py <n> <seed>"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_golden_classification as G  # noqa: E402

N, SEED = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(SEED)
cases, expected = [], []
for _ in range(N):
    c = G.make_case(rng, big=True)
    d = {"kind": G.TASKS[c["task"]], "preds": c["preds"], "target": c["target"]}
    d.update({f"p_{k}": float(v) for k, v in c["prm"].items()})
    if c["thrv"] is not None:
        d["thrv"] = c["thrv"]
    cases.append(d)
    expected.append(G.expected(c))
torch.save(cases, "cases.pt")
torch.save(expected, "expected.pt")
print(len(cases), "cases,", sum(len(e) for e in expected), "outputs")
