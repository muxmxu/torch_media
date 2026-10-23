"""Compares results.pt (metric_sweep) with expected.pt (sweep_gen.py) in the current directory."""
import collections, sys, torch
exp = torch.load("expected.pt"); got = torch.load("results.pt"); cases = torch.load("cases.pt")
stat = collections.defaultdict(lambda: [0, 0, 0.0])  # n, exact, max |delta|
for c, e, g in zip(cases, exp, got):
    kind = c["kind"]
    for k, v in e.items():
        name = f"{kind}{'_segm' if kind == 'map' and int(c['segm']) else ''}:{k}"
        s = stat[name]
        s[0] += 1
        if k not in g:
            print("missing", name); continue
        a, b = g[k].to(torch.float64).reshape(-1), v.to(torch.float64).reshape(-1)
        if a.shape != b.shape:
            print("shape", name, a.shape, b.shape); continue
        same = torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0)) and torch.equal(a.isnan(), b.isnan())
        s[1] += same
        fin = a.isfinite() & b.isfinite()
        if fin.any():
            s[2] = max(s[2], (a - b).abs()[fin].max().item())
for k, (n, ex, d) in sorted(stat.items()):
    print(f"{k:16s} {ex}/{n} bit-exact, max|delta| {d:.3g}")
