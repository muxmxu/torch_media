#!/usr/bin/env python3
"""Golden values for the torchmedia::metric::classification tests (develop_log/2026-10-24/progress02).

Run in the project venv; it rewrites golden_classification.hpp next to it:
    .venv/bin/python unit_test/metric/gen_golden_classification.py

90 cases are random (task, inputs, options) draws from make_case(), 7 more are hand-made (targeted(): option
combinations the random draws left unpinned). make_case() draws binary / multiclass / multilabel inputs with
probabilities (ties on a 0.05 grid), logits or labels, extra dims, ignore_index entries, NaN scores, all-same
predictions, degenerate targets (one class, absent classes), and random options (threshold, average, multidim_average,
top_k, zero_division, beta, thresholds as None / count / list / tensor, normalize, weights, max_fpr, fpr_range, the
fixed-point minimum, n_bins, norm, squared, multiclass_mode). expected() evaluates every metric of the task with
torchmetrics 1.9.0 — the functional on the whole input ("<family>/f/<metric>") and the class updated with the two
halves ("<family>/c/<metric>"); a curve's tuple / list elements get ".<i>" suffixes and a torchmetrics exception is
recorded as "<key>!". unit_test/metric/classification_driver.hpp evaluates the same keys with torchmedia;
sweep/sweep_classification_gen.py reuses make_case / expected for the randomised sweep. Float32 values are printed
with 9 significant digits, others with repr (both round trip exactly). Generated on arm64.
"""
import pathlib
import sys
import warnings

import numpy as np
import torch
import torchmetrics.classification as C
import torchmetrics.functional.classification as F
from torchmetrics.utilities.data import _squeeze_if_scalar

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).parent
TASKS = ("binary", "multiclass", "multilabel")


def rnd_probs(rng, shape, kind):
    if kind == "grid":  # ties
        return np.floor(rng.random(shape) * 20) / 20
    if kind == "logit":
        return rng.standard_normal(shape) * 3
    return rng.random(shape)


def make_case(rng, big=False):
    """One random case: dict(task, preds, target, thrv (thresholds values or None), prm={name: number}); big (sweep):
    2% of the cases have 3000-20000 samples (torchmetrics' loop updates beyond 50000 binary / 1e6 multiclass values)."""
    task = int(rng.integers(0, 3))
    n = int(rng.integers(3000, 20000)) if big and rng.random() < 0.02 else int(rng.integers(1, 25))
    extra = [int(rng.integers(2, 4))] if rng.random() < 0.35 else []
    c = int(rng.integers(2, 6))
    dt = [torch.float32, torch.float32, torch.float32, torch.float64, torch.float16][int(rng.integers(0, 5))]
    pkind = ["grid", "uniform", "logit", "labels"][int(rng.integers(0, 4))]
    prm = dict(task=task, C=c)
    # targets
    if task == 1:
        tshape = [n] + extra
        k = c if rng.random() < 0.7 else max(1, c - 1 - int(rng.integers(0, 2)))  # absent classes
        target = rng.integers(0, k, tshape)
        if rng.random() < 0.1:
            target[:] = int(rng.integers(0, c))
        if pkind == "labels":
            preds = torch.from_numpy(np.where(rng.random(tshape) < 0.6, target, rng.integers(0, c, tshape)))
        else:
            preds = torch.from_numpy(rnd_probs(rng, [n, c] + extra, pkind)).to(dt)
            if pkind == "uniform" and rng.random() < 0.5:
                preds = torch.softmax(preds.double() * 3, 1).to(dt)  # proper probabilities
    else:
        tshape = [n] + ([c] if task == 2 else []) + extra
        p1 = rng.random()
        target = (rng.random(tshape) < p1).astype(np.int64)
        if rng.random() < 0.1:
            target[:] = int(rng.integers(0, 2))
        if pkind == "labels":
            preds = torch.from_numpy(np.where(rng.random(tshape) < 0.7, target, 1 - target))
        else:
            preds = torch.from_numpy(rnd_probs(rng, tshape, pkind)).to(dt)
    if rng.random() < 0.05:  # all-same predictions
        preds = torch.full_like(preds, preds.reshape(-1)[0].item())
    if pkind != "labels" and rng.random() < 0.04:
        preds.view(-1)[int(rng.integers(0, preds.numel()))] = float("nan")
    ii = [None, -1, 255, 0, c - 1][int(rng.integers(0, 5))] if rng.random() < 0.45 else None
    if task != 1 and ii is not None and ii in (0, c - 1) and rng.random() < 0.7:
        ii = -1
    if ii is not None:
        target = np.where(rng.random(target.shape) < 0.15, ii, target)
        prm["ii"] = ii
    target = torch.from_numpy(np.asarray(target, dtype=np.int64))
    prm["thr"] = float([0.5, 0.5, 0.3, 0.75, 0.0, 1.0][int(rng.integers(0, 6))])
    prm["md"] = int(rng.random() < 0.4)
    prm["avg"] = int(rng.integers(0, 4))
    prm["topk"] = int(rng.integers(1, c + 1)) if task == 1 and pkind != "labels" and rng.random() < 0.4 else 1
    prm["zd"] = float(rng.integers(0, 2))
    prm["beta"] = float([1.0, 0.5, 2.0, 1.7][int(rng.integers(0, 4))])
    prm["ncm"] = int(rng.integers(0, 4))
    prm["kw"] = int(rng.integers(0, 3))
    tk = [0, 0, 1, 2, 3][int(rng.integers(0, 5))]
    prm["tk"] = tk
    thrv = None
    if tk == 1:
        prm["tn"] = int(rng.integers(2, 12))
    elif tk == 2:
        thrv = torch.tensor(sorted(set(np.round(rng.random(int(rng.integers(1, 8))), 2).tolist()) | {0.0, 0.5}),
                            dtype=torch.float64)
    elif tk == 3:
        v = rng.random(int(rng.integers(1, 9)))
        thrv = torch.from_numpy(np.sort(v) if rng.random() < 0.8 else v).to(
            [torch.float32, torch.float64][int(rng.integers(0, 2))])
    prm["cavg"] = [0, 1, 3][int(rng.integers(0, 3))]
    prm["aavg"] = [0, 1, 2, 3][int(rng.integers(0 if task == 2 else 1, 4))]
    prm["lavg"] = [1, 3][int(rng.integers(0, 2))]
    if rng.random() < 0.4:
        prm["mfpr"] = float([0.1, 0.3, 0.5, 0.8, 1.0][int(rng.integers(0, 5))])
    prm["flo"], prm["fhi"] = [(0.001, 0.1), (0.0, 1.0), (0.05, 0.5), (0.2, 0.9)][int(rng.integers(0, 4))]
    prm["mn"] = float([0.0, 0.25, 0.5, 0.8, 1.0][int(rng.integers(0, 5))])
    prm["nb"] = int([1, 3, 15][int(rng.integers(0, 3))])
    prm["cen"] = int(rng.integers(0, 3))
    prm["sq"] = int(rng.integers(0, 2))
    prm["hmode"] = int(rng.integers(0, 2))
    return dict(task=task, preds=preds, target=target, thrv=thrv, prm=prm)


def fixed_case(task, preds, target, **prm):
    """a hand-made case: make_case's parameters with fixed defaults, overridden by prm"""
    p = dict(task=task, thr=0.5, md=0, avg=1, topk=1, zd=0.0, beta=1.0, ncm=0, kw=0, tk=0, cavg=3, aavg=1, lavg=1,
             flo=0.001, fhi=0.1, mn=0.5, nb=15, cen=0, sq=0, hmode=0)
    p.update(prm)
    return dict(task=task, preds=preds, target=torch.as_tensor(target, dtype=torch.int64), thrv=None, prm=p)


def targeted():
    """configurations the 90 random draws left unpinned (mutants surviving them, progress02 Issues): multiclass micro
    global top_k 1, top_k 2 with a predicted class absent from the target, top_k 2 with ignore_index -1, multiclass
    jaccard micro and MultilabelJaccardIndex (functional vs class) with ignore_index inside [0, C)"""
    rng = np.random.default_rng(7)
    probs = lambda *shape: torch.from_numpy(rng.random(shape)).to(torch.float32)  # noqa: E731
    absent = probs(16, 4)
    absent[::2, 3] += 1.0  # class 3: top-1 in half the samples, never a target
    return [
        fixed_case(1, probs(16, 4), rng.integers(0, 4, 16), C=4, avg=0, zd=1.0),
        fixed_case(1, absent, rng.integers(0, 3, 16), C=4, avg=1, topk=2),
        fixed_case(1, probs(16, 4), np.where(rng.random(16) < 0.25, -1, rng.integers(0, 4, 16)), C=4, avg=1, topk=2,
                   ii=-1),
        fixed_case(1, probs(8, 4, 3), np.where(rng.random((8, 3)) < 0.25, -1, rng.integers(0, 4, (8, 3))), C=4,
                   avg=2, topk=2, md=1, ii=-1),
        fixed_case(1, probs(16, 4), np.where(rng.random(16) < 0.25, 1, rng.integers(0, 4, 16)), C=4, avg=0, ii=1),
        fixed_case(2, probs(16, 3), rng.integers(0, 2, (16, 3)), C=3, avg=0, ii=0),
        fixed_case(2, probs(16, 3), rng.integers(0, 2, (16, 3)), C=3, avg=1, ii=0),
    ]


AVG = {0: "micro", 1: "macro", 2: "weighted", 3: "none"}
NORM_CM = {0: None, 1: "true", 2: "pred", 3: "all"}
KW = {0: None, 1: "linear", 2: "quadratic"}
CE_NORM = {0: "l1", 1: "l2", 2: "max"}


def thresholds_of(case):
    prm = case["prm"]
    return {0: None, 1: prm.get("tn"), 2: None if case["thrv"] is None else case["thrv"].tolist(), 3: case["thrv"]}[
        prm["tk"]]


def flat(key, v, out):
    if isinstance(v, torch.Tensor):
        out[key] = v.detach().clone()
    else:
        for i, x in enumerate(v):
            flat(f"{key}.{i}", x, out)


def run(key, fn, out):
    try:
        flat(key, fn(), out)
    except Exception:
        out[key + "!"] = torch.tensor(1)


def split(x):
    k = max(1, x.shape[0] // 2)
    return x[:k], x[k:]


def validated(m):
    """torchmetrics' BinaryAUROC.__init__ never sets validate_args and BinaryLogAUC defaults it to False (their update
    checks nothing); torchmedia's classes check their inputs like the functionals (progress02 D7)"""
    m.validate_args = True
    return m


def f64_state(m, case):
    """torchmedia's float64 class sums for float64 inputs (progress11 D5): torchmetrics' float32 hinge state as float64"""
    if case["preds"].dtype == torch.float64:
        m.measures = m.measures.double()
    return m


def micro_binned_class(name, case, c, th, ii):
    """torchmetrics' Multiclass{PrecisionRecallCurve,ROC,EER} class with average micro and binned thresholds, its
    [T, C, 2, 2] state replaced by the [T, 2, 2] one its update produces (torchmedia's class; progress02 D7)"""
    from torchmetrics.functional.classification.eer import _eer_compute
    from torchmetrics.functional.classification.precision_recall_curve import (
        _adjust_threshold_arg, _multiclass_precision_recall_curve_compute, _multiclass_precision_recall_curve_format,
        _multiclass_precision_recall_curve_tensor_validation, _multiclass_precision_recall_curve_update)
    from torchmetrics.functional.classification.roc import _multiclass_roc_compute
    thr, state = _adjust_threshold_arg(th), 0
    for ph, tt in zip(split(case["preds"]), split(case["target"])):
        if ph.shape[0]:
            _multiclass_precision_recall_curve_tensor_validation(ph, tt, c, ii)
            ph, tt, _ = _multiclass_precision_recall_curve_format(ph, tt, c, thr, ii, "micro")
            state = state + _multiclass_precision_recall_curve_update(ph, tt, c, thr, "micro")
    if name == "pr":
        return _squeeze_if_scalar(_multiclass_precision_recall_curve_compute(state, c, thr, "micro"))
    fpr, tpr, t = _multiclass_roc_compute(state, c, thr, "micro")
    return _squeeze_if_scalar(_eer_compute(fpr, tpr) if name == "eer" else (fpr, tpr, t))


def both(family, name, fn, cls, case, out, cls_value=None):
    """the functional on the whole input and the class updated with the two halves (cls_value: the expected class
    value where torchmetrics' class is broken)"""
    p, t = case["preds"], case["target"]
    run(f"{family}/f/{name}", lambda: fn(p, t), out)
    if cls_value:
        run(f"{family}/c/{name}", cls_value, out)
        return

    def cls_value():
        m = cls()
        (p1, p2), (t1, t2) = split(p), split(t)
        m.update(p1, t1)
        if p2.shape[0]:
            m.update(p2, t2)
        return m.compute()

    run(f"{family}/c/{name}", cls_value, out)


def expected(case):
    task, prm = case["task"], case["prm"]
    out = {}
    c, ii, thr = prm["C"], prm.get("ii"), prm["thr"]
    md = ["global", "samplewise"][prm["md"]]
    avg, zd, beta = AVG[prm["avg"]], prm["zd"], prm["beta"]
    th = thresholds_of(case)
    mn, fr = prm["mn"], (prm["flo"], prm["fhi"])
    if task == 0:
        so = dict(threshold=thr, multidim_average=md, ignore_index=ii)
        for name, f, k, extra in (("stat_scores", F.binary_stat_scores, C.BinaryStatScores, {}),
                                  ("accuracy", F.binary_accuracy, C.BinaryAccuracy, {}),
                                  ("precision", F.binary_precision, C.BinaryPrecision, dict(zero_division=zd)),
                                  ("recall", F.binary_recall, C.BinaryRecall, dict(zero_division=zd)),
                                  ("f1", F.binary_f1_score, C.BinaryF1Score, dict(zero_division=zd)),
                                  ("specificity", F.binary_specificity, C.BinarySpecificity, {}),
                                  ("npv", F.binary_negative_predictive_value, C.BinaryNegativePredictiveValue,
                                   dict(zero_division=zd)),
                                  ("hamming", F.binary_hamming_distance, C.BinaryHammingDistance, {})):
            both("stat", name, lambda p, t, f=f, e=extra: f(p, t, **so, **e), lambda k=k, e=extra: k(**so, **e), case,
                 out)
        both("stat", "fbeta", lambda p, t: F.binary_fbeta_score(p, t, beta, **so, zero_division=zd),
             lambda: C.BinaryFBetaScore(beta, **so, zero_division=zd), case, out)
        co = dict(threshold=thr, ignore_index=ii)
        both("stat", "confmat", lambda p, t: F.binary_confusion_matrix(p, t, normalize=NORM_CM[prm["ncm"]], **co),
             lambda: C.BinaryConfusionMatrix(normalize=NORM_CM[prm["ncm"]], **co), case, out)
        both("stat", "jaccard", lambda p, t: F.binary_jaccard_index(p, t, zero_division=zd, **co),
             lambda: C.BinaryJaccardIndex(zero_division=zd, **co), case, out)
        both("stat", "mcc", lambda p, t: F.binary_matthews_corrcoef(p, t, **co), lambda: C.BinaryMatthewsCorrCoef(**co),
             case, out)
        both("stat", "kappa", lambda p, t: F.binary_cohen_kappa(p, t, weights=KW[prm["kw"]], **co),
             lambda: C.BinaryCohenKappa(weights=KW[prm["kw"]], **co), case, out)
        ko = dict(thresholds=th, ignore_index=ii)
        mf = prm.get("mfpr")
        for name, f, k, extra in (("pr", F.binary_precision_recall_curve, C.BinaryPrecisionRecallCurve, {}),
                                  ("roc", F.binary_roc, C.BinaryROC, {}),
                                  ("auroc", F.binary_auroc, lambda **k: validated(C.BinaryAUROC(**k)),
                                   dict(max_fpr=mf)),
                                  ("ap", F.binary_average_precision, C.BinaryAveragePrecision, {}),
                                  ("logauc", F.binary_logauc, lambda **k: validated(C.BinaryLogAUC(**k)),
                                   dict(fpr_range=fr)),
                                  ("eer", F.binary_eer, C.BinaryEER, {}),
                                  ("par", F.binary_precision_at_fixed_recall, C.BinaryPrecisionAtFixedRecall,
                                   dict(min_recall=mn)),
                                  ("rap", F.binary_recall_at_fixed_precision, C.BinaryRecallAtFixedPrecision,
                                   dict(min_precision=mn)),
                                  ("sas", F.binary_sensitivity_at_specificity, C.BinarySensitivityAtSpecificity,
                                   dict(min_specificity=mn)),
                                  ("spas", F.binary_specificity_at_sensitivity, C.BinarySpecificityAtSensitivity,
                                   dict(min_sensitivity=mn))):
            both("curve", name, lambda p, t, f=f, e=extra: f(p, t, **ko, **e), lambda k=k, e=extra: k(**ko, **e), case,
                 out)
        ceo = dict(n_bins=prm["nb"], norm=CE_NORM[prm["cen"]], ignore_index=ii)
        both("misc", "ce", lambda p, t: F.binary_calibration_error(p, t, **ceo), lambda: C.BinaryCalibrationError(**ceo),
             case, out)
        ho = dict(squared=bool(prm["sq"]), ignore_index=ii)
        both("misc", "hinge", lambda p, t: F.binary_hinge_loss(p, t, validate_args=True, **ho),
             lambda: f64_state(C.BinaryHingeLoss(**ho), case), case, out)
    elif task == 1:
        so = dict(num_classes=c, average=avg, top_k=prm["topk"], multidim_average=md, ignore_index=ii)
        for name, f, k, extra in (("stat_scores", F.multiclass_stat_scores, C.MulticlassStatScores, {}),
                                  ("accuracy", F.multiclass_accuracy, C.MulticlassAccuracy, {}),
                                  ("precision", F.multiclass_precision, C.MulticlassPrecision, dict(zero_division=zd)),
                                  ("recall", F.multiclass_recall, C.MulticlassRecall, dict(zero_division=zd)),
                                  ("f1", F.multiclass_f1_score, C.MulticlassF1Score, dict(zero_division=zd)),
                                  ("specificity", F.multiclass_specificity, C.MulticlassSpecificity, {}),
                                  ("npv", F.multiclass_negative_predictive_value, C.MulticlassNegativePredictiveValue,
                                   dict(zero_division=zd)),
                                  ("hamming", F.multiclass_hamming_distance, C.MulticlassHammingDistance, {})):
            both("stat", name, lambda p, t, f=f, e=extra: f(p, t, **so, **e), lambda k=k, e=extra: k(**so, **e), case,
                 out)
        both("stat", "fbeta", lambda p, t: F.multiclass_fbeta_score(p, t, beta, **so, zero_division=zd),
             lambda: C.MulticlassFBetaScore(beta, **so, zero_division=zd), case, out)
        eo = dict(num_classes=c, multidim_average=md, ignore_index=ii)
        both("stat", "exact", lambda p, t: F.multiclass_exact_match(p, t, **eo), lambda: C.MulticlassExactMatch(**eo),
             case, out)
        co = dict(num_classes=c, ignore_index=ii)
        both("stat", "confmat", lambda p, t: F.multiclass_confusion_matrix(p, t, normalize=NORM_CM[prm["ncm"]], **co),
             lambda: C.MulticlassConfusionMatrix(normalize=NORM_CM[prm["ncm"]], **co), case, out)
        both("stat", "jaccard", lambda p, t: F.multiclass_jaccard_index(p, t, average=avg, zero_division=zd, **co),
             lambda: C.MulticlassJaccardIndex(average=avg, zero_division=zd, **co), case, out)
        both("stat", "mcc", lambda p, t: F.multiclass_matthews_corrcoef(p, t, **co),
             lambda: C.MulticlassMatthewsCorrCoef(**co), case, out)
        both("stat", "kappa", lambda p, t: F.multiclass_cohen_kappa(p, t, weights=KW[prm["kw"]], **co),
             lambda: C.MulticlassCohenKappa(weights=KW[prm["kw"]], **co), case, out)
        cavg = None if prm["cavg"] == 3 else AVG[prm["cavg"]]
        ko = dict(num_classes=c, thresholds=th, ignore_index=ii)
        aavg, lavg = AVG[prm["aavg"]], AVG[prm["lavg"]]
        for name, f, k, extra in (("pr", F.multiclass_precision_recall_curve, C.MulticlassPrecisionRecallCurve,
                                   dict(average=cavg)),
                                  ("roc", F.multiclass_roc, C.MulticlassROC, dict(average=cavg)),
                                  ("auroc", F.multiclass_auroc, C.MulticlassAUROC, dict(average=aavg)),
                                  ("ap", F.multiclass_average_precision, C.MulticlassAveragePrecision,
                                   dict(average=aavg)),
                                  ("logauc", F.multiclass_logauc, C.MulticlassLogAUC, dict(average=lavg, fpr_range=fr)),
                                  ("eer", F.multiclass_eer, C.MulticlassEER, dict(average=cavg)),
                                  ("par", F.multiclass_precision_at_fixed_recall, C.MulticlassPrecisionAtFixedRecall,
                                   dict(min_recall=mn)),
                                  ("rap", F.multiclass_recall_at_fixed_precision, C.MulticlassRecallAtFixedPrecision,
                                   dict(min_precision=mn)),
                                  ("sas", F.multiclass_sensitivity_at_specificity,
                                   C.MulticlassSensitivityAtSpecificity, dict(min_specificity=mn)),
                                  ("spas", F.multiclass_specificity_at_sensitivity,
                                   C.MulticlassSpecificityAtSensitivity, dict(min_sensitivity=mn))):
            # torchmetrics' multiclass PR-curve classes keep a [T, C, 2, 2] state for average micro too, so their
            # update fails (or, for T == C, broadcasts into the wrong cells)
            broken = name in ("pr", "roc", "eer") and cavg == "micro" and th is not None
            both("curve", name, lambda p, t, f=f, e=extra: f(p, t, **ko, **e), lambda k=k, e=extra: k(**ko, **e), case,
                 out, (lambda name=name: micro_binned_class(name, case, c, th, ii)) if broken else None)
        ceo = dict(num_classes=c, n_bins=prm["nb"], norm=CE_NORM[prm["cen"]], ignore_index=ii)
        both("misc", "ce", lambda p, t: F.multiclass_calibration_error(p, t, **ceo),
             lambda: C.MulticlassCalibrationError(**ceo), case, out)
        ho = dict(num_classes=c, squared=bool(prm["sq"]), multiclass_mode=["crammer-singer", "one-vs-all"][prm["hmode"]],
                  ignore_index=ii)
        both("misc", "hinge", lambda p, t: F.multiclass_hinge_loss(p, t, validate_args=True, **ho),
             lambda: f64_state(C.MulticlassHingeLoss(**ho), case), case, out)
    else:
        so = dict(num_labels=c, threshold=thr, average=avg, multidim_average=md, ignore_index=ii)
        for name, f, k, extra in (("stat_scores", F.multilabel_stat_scores, C.MultilabelStatScores, {}),
                                  ("accuracy", F.multilabel_accuracy, C.MultilabelAccuracy, {}),
                                  ("precision", F.multilabel_precision, C.MultilabelPrecision, dict(zero_division=zd)),
                                  ("recall", F.multilabel_recall, C.MultilabelRecall, dict(zero_division=zd)),
                                  ("f1", F.multilabel_f1_score, C.MultilabelF1Score, dict(zero_division=zd)),
                                  ("specificity", F.multilabel_specificity, C.MultilabelSpecificity, {}),
                                  ("npv", F.multilabel_negative_predictive_value, C.MultilabelNegativePredictiveValue,
                                   dict(zero_division=zd)),
                                  ("hamming", F.multilabel_hamming_distance, C.MultilabelHammingDistance, {})):
            both("stat", name, lambda p, t, f=f, e=extra: f(p, t, **so, **e), lambda k=k, e=extra: k(**so, **e), case,
                 out)
        both("stat", "fbeta", lambda p, t: F.multilabel_fbeta_score(p, t, beta, **so, zero_division=zd),
             lambda: C.MultilabelFBetaScore(beta, **so, zero_division=zd), case, out)
        eo = dict(num_labels=c, threshold=thr, multidim_average=md, ignore_index=ii)
        both("stat", "exact", lambda p, t: F.multilabel_exact_match(p, t, **eo), lambda: C.MultilabelExactMatch(**eo),
             case, out)
        co = dict(num_labels=c, threshold=thr, ignore_index=ii)
        both("stat", "confmat", lambda p, t: F.multilabel_confusion_matrix(p, t, normalize=NORM_CM[prm["ncm"]], **co),
             lambda: C.MultilabelConfusionMatrix(normalize=NORM_CM[prm["ncm"]], **co), case, out)
        both("stat", "jaccard", lambda p, t: F.multilabel_jaccard_index(p, t, average=avg, zero_division=zd, **co),
             lambda: C.MultilabelJaccardIndex(average=avg, zero_division=zd, **co), case, out)
        both("stat", "mcc", lambda p, t: F.multilabel_matthews_corrcoef(p, t, **co),
             lambda: C.MultilabelMatthewsCorrCoef(**co), case, out)
        ko = dict(num_labels=c, thresholds=th, ignore_index=ii)
        aavg, lavg = AVG[prm["aavg"]], AVG[prm["lavg"]]
        for name, f, k, extra in (("pr", F.multilabel_precision_recall_curve, C.MultilabelPrecisionRecallCurve, {}),
                                  ("roc", F.multilabel_roc, C.MultilabelROC, {}),
                                  ("auroc", F.multilabel_auroc, C.MultilabelAUROC, dict(average=aavg)),
                                  ("ap", F.multilabel_average_precision, C.MultilabelAveragePrecision,
                                   dict(average=aavg)),
                                  ("logauc", F.multilabel_logauc, C.MultilabelLogAUC, dict(average=lavg, fpr_range=fr)),
                                  ("eer", F.multilabel_eer, C.MultilabelEER, {}),
                                  ("par", F.multilabel_precision_at_fixed_recall, C.MultilabelPrecisionAtFixedRecall,
                                   dict(min_recall=mn)),
                                  ("rap", F.multilabel_recall_at_fixed_precision, C.MultilabelRecallAtFixedPrecision,
                                   dict(min_precision=mn)),
                                  ("sas", F.multilabel_sensitivity_at_specificity,
                                   C.MultilabelSensitivityAtSpecificity, dict(min_specificity=mn)),
                                  ("spas", F.multilabel_specificity_at_sensitivity,
                                   C.MultilabelSpecificityAtSensitivity, dict(min_sensitivity=mn))):
            both("curve", name, lambda p, t, f=f, e=extra: f(p, t, **ko, **e), lambda k=k, e=extra: k(**ko, **e), case,
                 out)
        ro = dict(num_labels=c, ignore_index=ii)
        for name, f, k in (("coverage", F.multilabel_coverage_error, C.MultilabelCoverageError),
                           ("rank_ap", F.multilabel_ranking_average_precision, C.MultilabelRankingAveragePrecision),
                           ("rank_loss", F.multilabel_ranking_loss, C.MultilabelRankingLoss)):
            both("misc", name, lambda p, t, f=f: f(p, t, **ro), lambda k=k: k(**ro), case, out)
    return out


# ---------------------------------------------------------------- header writer
# The cases go into the header as text (a table of std::map initialisers took minutes to compile), parsed once by
# classification_test_util.hpp: "case <name>", "in|out <key> <dtype> <ndim> <shape...> <n> <values...>",
# "prm <key> <value>", "err <key>" (torchmetrics raised), split into raw-string chunks of < 60000 characters.
def fmt_values(t):
    t = torch.as_tensor(t).detach().cpu()
    if t.dtype == torch.float32:
        vals = [f"{v:.9g}" for v in t.flatten().tolist()]
    elif t.is_floating_point():
        vals = [repr(float(v)) for v in t.flatten().tolist()]
    else:
        vals = [str(int(v)) for v in t.flatten().tolist()]
    dtype = str(t.dtype).removeprefix("torch.")
    return " ".join([dtype, str(t.dim())] + [str(s) for s in t.shape] + [str(len(vals))] + vals)


def fmt_case(name, case, out):
    lines = [f"case {name}"]
    ins = {"preds": case["preds"], "target": case["target"]}
    if case["thrv"] is not None:
        ins["thrv"] = case["thrv"]
    lines += [f"in {k} {fmt_values(v)}" for k, v in ins.items()]
    lines += [f"prm {k} {repr(float(v))}" for k, v in case["prm"].items()]
    lines += [f"err {k[:-1]}" if k.endswith("!") else f"out {k} {fmt_values(v)}" for k, v in sorted(out.items())]
    return "\n".join(lines) + "\n"


def write(cases):
    text = "".join(fmt_case(f"case{i}", c, o) for i, (c, o) in enumerate(cases))
    chunks, cur = [], ""
    for line in text.splitlines(keepends=True):
        if len(cur) + len(line) > 60000:
            chunks.append(cur)
            cur = ""
        cur += line
    chunks.append(cur)
    body = ",\n".join(f'R"golden({c})golden"' for c in chunks)
    header = f"""// Generated by gen_golden_classification.py from torchmetrics 1.9.0 / torch 2.5.1 on arm64 -- do not edit.
// {len(cases)} cases as text (format: gen_golden_classification.py), parsed by classification_test_util.hpp.
// clang-format off
#pragma once
#include <vector>

namespace golden {{
    inline const std::vector<const char *> cls_text = {{
{body}}};
}} // namespace golden
"""
    (HERE / "golden_classification.hpp").write_text(header)


def main():
    rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 20261024)
    cases = []
    for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 90):
        case = make_case(rng)
        cases.append((case, expected(case)))
    cases += [(case, expected(case)) for case in targeted()]
    write(cases)
    print(len(cases), "cases,", sum(len(o) for _, o in cases), "outputs,",
          sum(k.endswith("!") for _, o in cases for k in o), "raised")


if __name__ == "__main__":
    main()
