"""The numpy VOC devkit implementation of gen_golden.py (voc_eval_np), importable without generating goldens."""
import numpy as np, torch
import torchvision.ops as ops
def voc_ap_np(rec, prec, use_07_metric):
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = 0 if np.sum(rec >= t) == 0 else np.max(prec[rec >= t])
            ap = ap + p / 11.0
        return ap
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


def voc_eval_np(preds, target, ovthresh=0.5, use_07_metric=False, pixel_inclusive=True, box_format="xyxy"):
    off = 1.0 if pixel_inclusive else 0.0
    xyxy = lambda b: ops.box_convert(b, box_format, "xyxy").double().numpy() if b.numel() else np.zeros((0, 4))
    P = [(xyxy(p["boxes"]), p["scores"].double().numpy(), p["labels"].numpy()) for p in preds]
    T = [(xyxy(t["boxes"]), t["labels"].numpy(), t.get("iscrowd", torch.zeros_like(t["labels"])).numpy()) for t in target]
    classes = sorted(set(np.concatenate([p[2] for p in P] + [t[1] for t in T]).tolist()))
    aps = []
    for c in classes:
        recs = {}
        npos = 0
        for i, (bb, lab, dif) in enumerate(T):
            sel = lab == c
            recs[i] = {"bbox": bb[sel], "difficult": dif[sel].astype(bool), "det": [False] * int(sel.sum())}
            npos += int((~dif[sel].astype(bool)).sum())
        image_ids = [i for i, p in enumerate(P) for l in p[2] if l == c]
        confidence = np.array([s for p in P for s, l in zip(p[1], p[2]) if l == c])
        BB = np.array([b for p in P for b, l in zip(p[0], p[2]) if l == c]).reshape(-1, 4)
        if npos == 0:
            aps.append(-1.0)
            continue
        sorted_ind = np.argsort(-confidence, kind="mergesort")
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R["bbox"].astype(float)
            if BBGT.size > 0:
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + off, 0.0)
                ih = np.maximum(iymax - iymin + off, 0.0)
                inters = iw * ih
                uni = ((bb[2] - bb[0] + off) * (bb[3] - bb[1] + off) +
                       (BBGT[:, 2] - BBGT[:, 0] + off) * (BBGT[:, 3] - BBGT[:, 1] + off) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
            if ovmax > ovthresh:
                if not R["difficult"][jmax]:
                    if not R["det"][jmax]:
                        tp[d] = 1.0
                        R["det"][jmax] = 1
                    else:
                        fp[d] = 1.0
            else:
                fp[d] = 1.0
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        aps.append(float(voc_ap_np(rec, prec, use_07_metric)))
    valid = [a for a in aps if a != -1.0]
    return (float(np.mean(valid)) if valid else -1.0), aps, classes


