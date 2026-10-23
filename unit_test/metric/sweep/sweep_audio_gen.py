"""Randomised audio / text parity sweep (dev-time, not in ctest; develop_log/2026-10-24/progress01 D9): torchmetrics
SNR / SI-SDR / SI-SNR / C-SI-SNR / SA-SDR / SDR / PIT, pystoi STOI, mir_eval bss_eval_sources, sklearn-ROC EER /
minDCF and torchmetrics WER / CER / MER / WIL / WIP / EditDistance. Writes cases.pt and expected.pt for the metric_sweep
driver, like sweep_gen.py. From a scratch directory:
    .venv/bin/python <repo>/unit_test/metric/sweep/sweep_audio_gen.py 200 1   # cases.pt, expected.pt
    <build>/unit_test/metric/metric_sweep                                    # -> results.pt
    .venv/bin/python <repo>/unit_test/metric/sweep/compare.py
Usage: sweep_audio_gen.py <n> <seed>"""
import os, sys, warnings
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # threaded numpy solves change run to run (gen_golden.py)
import numpy as np, torch
import jiwer
import mir_eval.separation as MES
import sklearn.metrics
import torchmetrics.functional.audio as FA
import torchmetrics.functional.text as FT
warnings.filterwarnings("ignore")
N, SEED = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
cases, expected = [], []


def speech(n, fs):
    """speech-like: jittered pulse train through two random resonators, syllables with silent gaps."""
    period, syl = fs // int(rng.integers(90, 250)), max(fs // int(rng.integers(3, 8)), 1)
    a1, a2 = rng.uniform(0.8, 1.6), rng.uniform(0.6, 0.9)
    e = (rng.random(n) - 0.5) * 0.02
    e[np.cumsum(period + rng.integers(0, 8, n // period + 1))[: n // period] % n] += 1.0
    y = np.zeros(n); z = np.zeros(n)
    for t in range(n):
        y[t] = a1 * y[t - 1] - a2 * y[t - 2] + e[t] if t > 1 else e[t]
        z[t] = 0.4 * z[t - 1] - 0.6 * z[t - 2] + y[t] if t > 1 else y[t]
    k = np.arange(n) % syl
    env = np.where((np.arange(n) // syl) % 3 == 2, 0.0, (k / syl) * (1 - k / syl) * 4)
    return z * env


def t_(x, dt):
    return torch.from_numpy(np.ascontiguousarray(x)).to(dt)


for i in range(N):
    dt = torch.float64 if i % 2 else torch.float32
    # SNR family
    b, s, n = int(rng.integers(1, 4)), int(rng.integers(2, 5)), int(rng.integers(64, 3000))
    tgt = rng.standard_normal((b, s, n)) * rng.uniform(0.1, 3)
    prd = tgt * rng.uniform(0.2, 1.5) + rng.standard_normal((b, s, n)) * rng.uniform(0.01, 2)
    p, t = t_(prd, dt), t_(tgt, dt)
    zm = bool(rng.integers(0, 2))
    out = dict(snr=FA.signal_noise_ratio(p, t, zm), sisdr=FA.scale_invariant_signal_distortion_ratio(p, t, zm),
               sisnr=FA.scale_invariant_signal_noise_ratio(p, t),
               sasdr=FA.source_aggregated_signal_distortion_ratio(p, t, bool(i % 3), zm))
    if n % 2 == 0:
        out["csisnr"] = FA.complex_scale_invariant_signal_noise_ratio(p.reshape(b, s, n // 2, 2),
                                                                     t.reshape(b, s, n // 2, 2), zm)
    cases.append(dict(kind="snr", p=p, t=t, zm=torch.tensor(int(zm)), si=torch.tensor(int(bool(i % 3)))))
    expected.append(out)
    # SDR
    m = int(rng.integers(100, 2500))
    tgt = rng.standard_normal((2, m))
    prd = 0.8 * tgt + 0.3 * np.roll(tgt, int(rng.integers(1, 30)), axis=1) + 0.2 * rng.standard_normal((2, m))
    fl = int(rng.choice([16, 64, 128, 512]))
    cg = None if i % 3 else int(rng.integers(0, 20))
    ld = None if i % 4 else float(rng.uniform(1e-6, 1e-2))
    p, t = t_(prd, dt), t_(tgt, dt)
    cases.append(dict(kind="sdr", p=p, t=t, fl=torch.tensor(fl), cg=torch.tensor(-1 if cg is None else cg),
                      ld=torch.tensor(-1.0 if ld is None else ld, dtype=torch.float64), zm=torch.tensor(int(zm))))
    expected.append(dict(sdr=FA.signal_distortion_ratio(p, t, cg, fl, zm, ld)))
    # PIT
    spk = int(rng.integers(2, 6))
    tgt = rng.standard_normal((2, spk, 300))
    perm = rng.permutation(spk)
    prd = tgt[:, perm] + rng.standard_normal((2, spk, 300)) * rng.uniform(0.1, 3)
    if i % 5 == 0:
        tgt[:, 1] = tgt[:, 0]  # ties
    p, t = t_(prd, dt), t_(tgt, dt)
    fn = int(rng.integers(0, 4))
    funcs = [(FA.scale_invariant_signal_distortion_ratio, "speaker-wise", "max"),
             (FA.signal_noise_ratio, "speaker-wise", "max"),
             (lambda a, c: ((a - c) ** 2).mean(-1), "speaker-wise", "min"),
             (FA.source_aggregated_signal_distortion_ratio, "permutation-wise", "max")]
    f, mode, ev = funcs[fn]
    if mode == "permutation-wise" and spk > 4:
        spk = 4
        p, t = p[:, :4].contiguous(), t[:, :4].contiguous()
    bm, bp = FA.permutation_invariant_training(p, t, f, mode, ev)
    cases.append(dict(kind="pit", p=p, t=t, fn=torch.tensor(fn)))
    expected.append(dict(metric=bm, perm=bp))
    # bss_eval_sources
    ns, n = int(rng.integers(1, 4)), int(rng.integers(600, 3000))
    ref = np.stack([speech(n, 8000) + 1e-3 * rng.standard_normal(n) for _ in range(ns)])
    mix = rng.uniform(0, 0.5, (ns, ns)) + np.eye(ns)[rng.permutation(ns)]
    est = mix @ ref + 0.05 * rng.standard_normal((ns, n))
    cp = bool(i % 4)
    r = MES.bss_eval_sources(ref, est, compute_permutation=cp)
    cases.append(dict(kind="bss", r=t_(ref, torch.float64), e=t_(est, torch.float64), cp=torch.tensor(int(cp))))
    expected.append(dict(sdr=t_(r[0], torch.float64), sir=t_(r[1], torch.float64), sar=t_(r[2], torch.float64),
                         perm=torch.from_numpy(np.asarray(r[3]))))
    # speaker verification
    nt, nn = int(rng.integers(1, 400)), int(rng.integers(1, 1500))
    dec = int(rng.integers(0, 4))
    tar = np.round(rng.normal(rng.uniform(0, 3), 1, nt), dec)
    non = np.round(rng.normal(0, 1, nn), dec)
    fpr, tpr, _ = sklearn.metrics.roc_curve(np.r_[np.ones(nt), np.zeros(nn)], np.r_[tar, non], drop_intermediate=False)
    fnr = 1 - tpr
    d = fnr - fpr
    j = int(np.argmax(d <= 0))
    eer = fpr[j] if d[j] == 0 else fpr[j - 1] + d[j - 1] / (d[j - 1] - d[j]) * (fpr[j] - fpr[j - 1])
    pt, cm, cf = float(rng.choice([0.01, 0.05, 0.5])), float(rng.choice([1.0, 10.0])), float(rng.choice([1.0, 2.0]))
    dcf = np.min(cm * fnr * pt + cf * fpr * (1 - pt)) / min(cm * pt, cf * (1 - pt))
    cases.append(dict(kind="sv", tar=t_(tar, torch.float64), non=t_(non, torch.float64),
                      prm=torch.tensor([pt, cm, cf], dtype=torch.float64)))
    expected.append(dict(eer=torch.tensor(eer, dtype=torch.float64), dcf=torch.tensor(dcf, dtype=torch.float64)))

# STOI (fewer cases: the Python reference is slow)
for i in range(N // 4):
    fs = int(rng.choice([8000, 10000, 11025, 12000, 16000, 22050, 32000, 44100, 48000]))
    n = int(rng.integers(fs // 3, 2 * fs))
    x = speech(n, fs)
    y = x * rng.uniform(0.3, 2) + rng.standard_normal(n) * rng.uniform(0.01, 1)
    ext = bool(i % 2)
    p, t = t_(y, torch.float64), t_(x, torch.float64)
    cases.append(dict(kind="stoi", p=p, t=t, fs=torch.tensor(fs), ext=torch.tensor(int(ext))))
    expected.append(dict(stoi=FA.short_time_objective_intelligibility(p[None], t[None], fs, ext)))

# text: random words over ASCII, CJK, kana, emoji with random separators and edits
ALPHA = list("abcdefghij") + list("天气很好我爱北京") + list("テキストのです") + ["😀", "é"]
SEPS = [" ", " ", " ", "  ", "\t", "\n", "　", " ", " "]
jiwer_agree = [0, 0]
for i in range(N):
    def sentence(k):
        return [("".join(rng.choice(ALPHA, int(rng.integers(1, 4))))) for _ in range(k)]
    pr, tg = [], []
    for _ in range(int(rng.integers(1, 6))):
        w = sentence(int(rng.integers(0, 8)))
        v = [x for x in w if rng.random() > 0.2]
        v = [x if rng.random() > 0.2 else "".join(rng.choice(ALPHA, 2)) for x in v]
        if rng.random() < 0.5:
            v.insert(int(rng.integers(0, len(v) + 1)), "".join(rng.choice(ALPHA, 2)))
        join = lambda ws: "".join(x + str(rng.choice(SEPS)) for x in ws).strip(" ") if i % 2 else " ".join(ws)
        tg.append(join(w)); pr.append(join(v))
    sc = int(rng.integers(0, 3))
    out = dict(wer=FT.word_error_rate(pr, tg), cer=FT.char_error_rate(pr, tg), mer=FT.match_error_rate(pr, tg),
               wil=FT.word_information_lost(pr, tg), wip=FT.word_information_preserved(pr, tg),
               edit=FT.edit_distance(pr, tg, substitution_cost=sc, reduction="none"))
    if i % 2 == 0 and all(tg):  # jiwer tokenises like str.split() here (single spaces)
        jiwer_agree[0] += 1
        jiwer_agree[1] += abs(jiwer.wer(tg, pr) - float(out["wer"])) < 1e-6 and abs(jiwer.cer(tg, pr) - float(out["cer"])) < 1e-6
    cases.append(dict(kind="text", p=pr, t=tg, sc=torch.tensor(sc)))
    expected.append(out)
print("jiwer WER / CER equal to torchmetrics on", jiwer_agree[1], "of", jiwer_agree[0], "single-space corpora")
torch.save(cases, "cases.pt")
torch.save(expected, "expected.pt")
print(len(cases), "cases")
