"""Randomised parity sweep (dev-time, not in ctest): MeanAveragePrecision vs pycocotools COCOeval (float64), VOC vs the
numpy devkit implementation, and image / segmentation metrics vs torchmetrics. Writes cases.pt for sweep.cpp and
expected.pt in the current directory (develop_log/2026-10-23/progress11 D10). From a scratch directory:
    .venv/bin/python <repo>/unit_test/metric/sweep/sweep_gen.py 600 1      # cases.pt, expected.pt
    <build>/unit_test/metric/metric_sweep                                  # cases.pt -> results.pt (opt-in target)
    .venv/bin/python <repo>/unit_test/metric/sweep/compare.py               # bit-exact counts per metric
Usage: sweep_gen.py <n_map_cases> <seed>"""
import contextlib, io, pathlib, sys, warnings
import numpy as np, torch
import torchvision.ops as ops
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.helpers import CocoBackend, _get_classes, _get_safe_item_values
import torchmetrics.functional.image as FI
import torchmetrics.functional.segmentation as FS
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # gen_golden_voc
N, SEED = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
FMT = ["xyxy", "xywh", "cxcywh"]

def coco_stats(preds, target, box_format, iou_type, iou_thr, rec_thr, mdt, average, class_metrics):
    backend = CocoBackend("pycocotools")
    det = {"box": [], "mask": [], "labels": [], "scores": []}
    gt = {"box": [], "mask": [], "labels": [], "crowds": [], "area": []}
    for p in preds:
        b, m = _get_safe_item_values((iou_type,), box_format, mdt, backend, p)
        if b is not None: det["box"].append(b)
        if m is not None: det["mask"].append(m)
        det["labels"].append(p["labels"]); det["scores"].append(p["scores"])
    for t in target:
        b, m = _get_safe_item_values((iou_type,), box_format, mdt, backend, t)
        if b is not None: gt["box"].append(b)
        if m is not None: gt["mask"].append(m)
        gt["labels"].append(t["labels"]); gt["crowds"].append(t["iscrowd"]); gt["area"].append(t["area"])
    def run(avg, cats=None):
        cp, ct = backend._get_coco_datasets(gt["labels"], gt["box"], gt["mask"], gt["crowds"], gt["area"], det["labels"],
                                            det["box"], det["mask"], det["scores"], (iou_type,), average=avg)
        if len(cp.imgs) == 0 or len(ct.imgs) == 0:
            return None
        ev = COCOeval(ct, cp, iouType=iou_type)
        ev.params.iouThrs = np.array(iou_thr, dtype=np.float64); ev.params.recThrs = np.array(rec_thr, dtype=np.float64)
        ev.params.maxDets = mdt
        out = []
        for c in cats or [None]:
            if c is not None: ev.params.catIds = [c]
            with contextlib.redirect_stdout(io.StringIO()):
                ev.evaluate(); ev.accumulate(); ev.summarize()
            out.append(ev.stats.copy())
        return out
    res = run(average)
    stats = res[0] if res else np.full(12, -1.0)
    classes = _get_classes(det["labels"], gt["labels"])
    pc = run("macro", classes) if class_metrics and res else None
    mpc = torch.tensor([s[0] for s in pc]) if pc else torch.tensor([-1.0])
    rpc = torch.tensor([s[8] for s in pc]) if pc else torch.tensor([-1.0])
    return torch.tensor(stats, dtype=torch.float64), mpc.double(), rpc.double()

MH, MW = 24, 28
def make_mask(p):
    x0, y0, x1, y1, hole = p
    m = torch.zeros(MH, MW, dtype=torch.uint8); m[y0:y1, x0:x1] = 1
    if hole: m[y0 + (y1 - y0) // 3:y1 - (y1 - y0) // 3, x0 + (x1 - x0) // 3:x1 - (x1 - x0) // 3] = 0
    return m

def rand_case(segm):
    n_img = int(rng.integers(1, 40))
    labels = rng.choice(10, size=int(rng.integers(1, 6)), replace=False).tolist()
    crowd_p, area_p = rng.choice([0, 0.1, 0.4]), rng.choice([0, 0.3])
    tie_dec = int(rng.choice([1, 2, 3]))
    W, H = (MW, MH) if segm else (int(rng.choice([64, 640])), int(rng.choice([48, 480])))
    preds, target, flat = [], [], {k: [] for k in ["pb", "ps", "pl", "pc", "gb", "gl", "gcr", "ga", "gc"]}
    for _ in range(n_img):
        ng = 0 if rng.uniform() < 0.15 else int(rng.integers(1, 8))
        np_ = 0 if rng.uniform() < 0.15 else int(rng.integers(1, 14 if rng.uniform() < 0.9 else 130))
        if segm:
            def prm():
                x0, y0 = int(rng.integers(0, MW - 3)), int(rng.integers(0, MH - 3))
                return [x0, y0, int(rng.integers(x0 + 1, MW + 1)), int(rng.integers(y0 + 1, MH + 1)), int(rng.integers(0, 2))]
            gbox = [prm() for _ in range(ng)]
            pbox = [prm() if (rng.uniform() < 0.4 or not gbox) else (lambda q: [max(0, q[0] - 1), q[1], q[2], q[3], q[4]])(gbox[int(rng.integers(0, len(gbox)))]) for _ in range(np_)]
        else:
            def box(near=None):
                if near is not None and rng.uniform() < 0.7:
                    j = rng.normal(0, 0.15, 4) * np.array([near[2] - near[0], near[3] - near[1]] * 2)
                    b = np.array(near) + j
                    return [b[0], b[1], max(b[2], b[0] + 0.5), max(b[3], b[1] + 0.5)]
                s = rng.choice([rng.uniform(1, 31), rng.uniform(33, 95), rng.uniform(97, 400)]) * (W / 640)
                w, h = s * rng.uniform(0.6, 1.6), s * rng.uniform(0.6, 1.6)
                x, y = rng.uniform(0, max(1, W - w)), rng.uniform(0, max(1, H - h))
                return [x, y, x + w, y + h]
            gbox = [box() for _ in range(ng)]
            pbox = [box(gbox[int(rng.integers(0, ng))] if ng else None) for _ in range(np_)]
        gl = [int(rng.choice(labels)) for _ in range(ng)]
        gcr = [int(rng.uniform() < crowd_p) for _ in range(ng)]
        ga = [float(np.float32(rng.uniform(10, 5000))) if rng.uniform() < area_p else 0.0 for _ in range(ng)]
        pl = [int(rng.choice(labels)) for _ in range(np_)]
        ps = [round(float(rng.uniform()), tie_dec) for _ in range(np_)]
        if segm:
            pt = dict(masks=torch.stack([make_mask(b) for b in pbox]) if pbox else torch.zeros(0, MH, MW, dtype=torch.uint8))
            tt = dict(masks=torch.stack([make_mask(b) for b in gbox]) if gbox else torch.zeros(0, MH, MW, dtype=torch.uint8))
            pb, gb = torch.tensor(pbox, dtype=torch.float64).reshape(-1, 5), torch.tensor(gbox, dtype=torch.float64).reshape(-1, 5)
        else:
            pb, gb = torch.tensor(pbox, dtype=torch.float32).reshape(-1, 4), torch.tensor(gbox, dtype=torch.float32).reshape(-1, 4)
            pt, tt = dict(boxes=pb), dict(boxes=gb)
        pt.update(scores=torch.tensor(ps, dtype=torch.float32), labels=torch.tensor(pl, dtype=torch.int64))
        tt.update(labels=torch.tensor(gl, dtype=torch.int64), iscrowd=torch.tensor(gcr, dtype=torch.int64), area=torch.tensor(ga, dtype=torch.float32))
        preds.append(pt); target.append(tt)
        for k, v in zip(flat, [pb, pt["scores"], pt["labels"], torch.tensor([np_]), gb, tt["labels"], tt["iscrowd"], tt["area"], torch.tensor([ng])]):
            flat[k].append(v)
    return preds, target, {k: torch.cat(v) for k, v in flat.items()}

cases, expected = [], []
for i in range(N):
    segm = i % 6 == 5
    preds, target, flat = rand_case(segm)
    fmt = "xyxy" if segm else FMT[int(rng.integers(0, 3))]
    if not segm and fmt != "xyxy":
        preds = [{**p, "boxes": ops.box_convert(p["boxes"], "xyxy", fmt)} for p in preds]
        target = [{**t, "boxes": ops.box_convert(t["boxes"], "xyxy", fmt)} for t in target]
        flat["pb"] = ops.box_convert(flat["pb"], "xyxy", fmt); flat["gb"] = ops.box_convert(flat["gb"], "xyxy", fmt)
    r = rng.uniform()
    iou_thr = torch.linspace(0.5, 0.95, 10).tolist() if r < 0.7 else sorted(rng.uniform(0.3, 0.95, int(rng.integers(1, 6))).round(2).tolist())
    rec_thr = torch.linspace(0.0, 1.0, 101).tolist() if r < 0.8 else rng.uniform(0, 1, int(rng.integers(1, 12))).round(3).tolist()
    mdt = sorted([1, 10, 100] if rng.uniform() < 0.7 else rng.integers(1, 150, 3).tolist())
    average = "micro" if rng.uniform() < 0.2 else "macro"
    cm = bool(rng.uniform() < 0.5)
    stats, mpc, rpc = coco_stats(preds, target, fmt, "segm" if segm else "bbox", iou_thr, rec_thr, mdt, average, cm)
    case = dict(kind="map", segm=torch.tensor(int(segm)), fmt=torch.tensor(FMT.index(fmt)), iou=torch.tensor(iou_thr, dtype=torch.float64),
                rec=torch.tensor(rec_thr, dtype=torch.float64), mdt=torch.tensor(mdt), micro=torch.tensor(int(average == "micro")),
                cm=torch.tensor(int(cm)), **flat)
    cases.append(case); expected.append(dict(stats=stats, mpc=mpc, rpc=rpc))
    if not segm:  # VOC on the same data
        from gen_golden_voc import voc_eval_np
        thr, use07, pix = float(rng.choice([0.3, 0.5, 0.7])), bool(rng.uniform() < 0.5), bool(rng.uniform() < 0.7)
        m, aps, cls = voc_eval_np(preds, target, thr, use07, pix, fmt)
        cases.append(dict(kind="voc", fmt=torch.tensor(FMT.index(fmt)), thr=torch.tensor(thr, dtype=torch.float64), use07=torch.tensor(int(use07)),
                          pix=torch.tensor(int(pix)), **flat))
        expected.append(dict(stats=torch.tensor([m] + aps, dtype=torch.float64)))

# image metrics: random shapes / options / dtypes
def pair(shape, dtype):
    t = torch.rand(shape, dtype=torch.float64)
    p = (t + 0.2 * (torch.rand(shape, dtype=torch.float64) - 0.5)).clamp(0, 1)
    return p.to(dtype), t.to(dtype)
for i in range(N):
    dt = torch.float32 if i % 3 else torch.float64
    b, c = int(rng.integers(1, 4)), int(rng.integers(2, 5))
    h, w = int(rng.integers(20, 70)), int(rng.integers(20, 70))
    p, t = pair((b, c, h, w), dt)
    ks = int(rng.choice([3, 5, 7, 11])); sg = float(rng.choice([0.5, 1.0, 1.5, 2.3]))
    dr = [None, 1.0, (0.1, 0.9)][int(rng.integers(0, 3))]
    ws = int(rng.integers(1, min(h, w) // 2))
    ratio = float(rng.uniform(1, 5))
    out = dict(
        psnr=FI.peak_signal_noise_ratio(p, t, data_range=1.0 if dr is None else dr, dim=(1, 2, 3) if i % 2 else None, reduction="none" if i % 2 else "elementwise_mean"),
        ssim=FI.structural_similarity_index_measure(p, t, gaussian_kernel=bool(i % 4), sigma=sg, kernel_size=ks, data_range=dr, reduction="none"),
        uqi=FI.universal_image_quality_index(p, t, kernel_size=(ks, ks), sigma=(sg, sg), reduction="none"),
        tv=FI.total_variation(p, reduction="none"),
        sam=FI.spectral_angle_mapper(p, t, reduction="none"),
        ergas=FI.error_relative_global_dimensionless_synthesis(p, t, ratio=ratio, reduction="none"),
        rmse=FI.root_mean_squared_error_using_sliding_window(p, t, window_size=ws, return_rmse_map=True)[1],
        rase=FI.relative_average_spectral_error(p, t, window_size=ws),
        scc=FI.spatial_correlation_coefficient(p, t, window_size=min(ws + 1, h, w), reduction="none"),
    )
    if h >= 41 and w >= 41:
        out["vif"] = FI.visual_information_fidelity(p, t, reduction="none")
    if min(h, w) // 4 > max(ks, int(3.5 * sg + 0.5) * 2 + 1):
        out["msssim"] = FI.multiscale_structural_similarity_index_measure(p, t, kernel_size=ks, sigma=sg, betas=(0.4, 0.35, 0.25), data_range=dr, reduction="none")
    ergas_ratio = None
    cases.append(dict(kind="image", p=p, t=t, ks=torch.tensor(ks), sg=torch.tensor(sg, dtype=torch.float64), dr=torch.tensor([] if dr is None else ([dr] if isinstance(dr, float) else list(dr)), dtype=torch.float64),
                      ws=torch.tensor(ws), i=torch.tensor(i), ratio=torch.tensor(ratio, dtype=torch.float64)))
    expected.append(out)
# segmentation
for i in range(N // 2):
    n, c, h, w = int(rng.integers(1, 5)), int(rng.integers(2, 7)), int(rng.integers(3, 20)), int(rng.integers(3, 20))
    tgt = torch.from_numpy(rng.integers(0, c, (n, h, w)))
    prd = torch.where(torch.rand(n, h, w) < 0.5, tgt, torch.from_numpy(rng.integers(0, c, (n, h, w))))
    if i % 3 == 0:
        tgt = torch.clamp(tgt, max=c - 2)  # an absent class
    bg = bool(i % 2)
    out = dict(miou=FS.mean_iou(prd, tgt, num_classes=c, include_background=bg, per_class=True, input_format="index"),
               dice=FS.dice_score(prd, tgt, num_classes=c, include_background=bg, average="none", input_format="index"),
               dicew=FS.dice_score(prd, tgt, num_classes=c, include_background=bg, average="weighted", input_format="index"),
               gdice=FS.generalized_dice_score(prd, tgt, num_classes=c, include_background=bg, per_class=True, input_format="index"),
               gdices=FS.generalized_dice_score(prd, tgt, num_classes=c, include_background=bg, weight_type="simple", input_format="index"))
    cases.append(dict(kind="seg", p=prd, t=tgt, c=torch.tensor(c), bg=torch.tensor(int(bg))))
    expected.append(out)
torch.save(cases, "cases.pt")
torch.save(expected, "expected.pt")
print(len(cases), "cases")
