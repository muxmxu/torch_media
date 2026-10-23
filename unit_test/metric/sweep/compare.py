"""Compares results.pt (metric_sweep) with expected.pt (sweep_gen.py) in the current directory.

Per metric: entries bit-exact (same dtype, same shape, same values with NaN == NaN), entries whose values are equal but
whose dtype or shape differs, and the largest finite |delta|; then the totals. A "<key>!" entry (classification: raised)
counts as exact when both sides raised."""
import collections, re, sys, torch
exp = torch.load("expected.pt"); got = torch.load("results.pt"); cases = torch.load("cases.pt")
stat = collections.defaultdict(lambda: [0, 0, 0, 0.0])  # n, exact, values equal but dtype / shape differ, max |delta|
tot = collections.Counter()
for c, e, g in zip(cases, exp, got):
    kind = c["kind"]
    tot["extra"] += len(set(g) - set(e))
    for k, v in e.items():
        # classification keys: per-class list elements ("<key>.<i>.<j>") and "<key>!" (raised) grouped by metric
        base = re.sub(r"(\.\d+)+$", "", k)
        name = f"{kind}{'_segm' if kind == 'map' and int(c['segm']) else ''}:{base}"
        s = stat[name]
        s[0] += 1
        tot["n"] += 1
        if k not in g:
            print("missing", name); tot["mismatch"] += 1; continue
        if k.endswith("!"):
            s[1] += 1; tot["raised"] += 1; continue
        layout = g[k].dtype == v.dtype and g[k].shape == v.shape
        a, b = g[k].to(torch.float64).reshape(-1), v.to(torch.float64).reshape(-1)
        if a.shape != b.shape:
            print("shape", name, g[k].shape, v.shape); tot["mismatch"] += 1; continue
        same = torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0)) and torch.equal(a.isnan(), b.isnan())
        s[1] += same and layout
        s[2] += same and not layout
        tot["exact" if same and layout else "layout" if same else "mismatch"] += 1
        fin = a.isfinite() & b.isfinite()
        if fin.any():
            s[3] = max(s[3], (a - b).abs()[fin].max().item())
for k, (n, ex, lay, d) in sorted(stat.items()):
    print(f"{k:16s} {ex}/{n} bit-exact, {lay} dtype/shape differ, max|delta| {d:.3g}")
print(f"cases {len(cases)}; outputs {tot['n']}: values bit-exact (dtype, shape, NaN) {tot['exact']}, raised on both "
      f"{tot['raised']}, equal values but dtype / shape differ {tot['layout']}, mismatches {tot['mismatch']}, "
      f"extra keys {tot['extra']}")
