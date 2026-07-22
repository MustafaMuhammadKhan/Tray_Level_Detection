

import os
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TRAY_MODEL_PATH   = r"C:\Users\Lenovo\Desktop\SweetDetection\Models\best.pt"
EMPTY_MODEL_PATH  = r"C:\Users\Lenovo\Desktop\SweetDetection\Models\empty_best.pt"
VIDEO_PATH        = r"C:\Users\Lenovo\Desktop\SweetDetection\TestVideo.mp4"
OUTPUT_VIDEO      = r"C:\Users\Lenovo\Desktop\SweetDetection\Result.mp4"
DEBUG_FOLDER      = r"C:\Users\Lenovo\Desktop\SweetDetection\debug_images"

CONF_TRAY         = 0.40
CONF_EMPTY        = 0.25

ALERT_THRESH      = 0.55   
HALF_THRESH       = 0.15   

SMOOTH_WINDOW     = 15     
EMA_ALPHA         = 0.15   
HYSTERESIS_MARGIN = 0.10

CONF_WEIGHT_FLOOR = 0.40   
MORPH_KERNEL_SIZE = 7      
TRAY_ERODE_SIZE   = 5     

STALE_AFTER       = 60
DEBUG_FRAMES      = 5

TRACKER_CFG       = "bytetrack.yaml"
DEVICE            = None
IMG_SIZE_TRAY     = 640
IMG_SIZE_EMPTY    = 640

TRAY_SKIP         = 2
EMPTY_SKIP        = 10

os.makedirs(DEBUG_FOLDER, exist_ok=True)

COLOURS = {
    "Full":  (0, 200,   0),
    "Half":  (0, 165, 255),
    "Empty": (0,   0, 255),
}


_MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
_ERODE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (TRAY_ERODE_SIZE, TRAY_ERODE_SIZE))



def build_empty_mask(empty_results, h, w):
    
    weight_map = np.zeros((h, w), dtype=np.float32)

    if empty_results is not None and empty_results.masks is not None:
        confs = (empty_results.boxes.conf.cpu().numpy()
                 if empty_results.boxes is not None else [])

        for i, seg in enumerate(empty_results.masks.xy):
            pts = np.round(seg).astype(np.int32)
            if pts.shape[0] < 3:
                continue
            weight = float(confs[i]) if i < len(confs) else 1.0
            cv2.fillPoly(weight_map, [pts], weight)

    
    binary = (weight_map > CONF_WEIGHT_FLOOR).astype(np.uint8) * 255

    
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    return binary


def compute_ratio(tray_pts, empty_mask, h, w):
    
    tray_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(tray_mask, [tray_pts], 255)

    
    tray_eroded = cv2.erode(tray_mask, _ERODE_KERNEL, iterations=1)

    tray_px = int(np.sum(tray_eroded == 255))
    if tray_px == 0:
        return 0.0, tray_mask

    inter_px = int(np.sum(cv2.bitwise_and(tray_eroded, empty_mask) == 255))
    ratio    = float(np.clip(inter_px / tray_px, 0.0, 1.0))
    return ratio, tray_mask   


def classify(ratio):
    if ratio >= ALERT_THRESH:
        return "Empty"
    if ratio >= HALF_THRESH:
        return "Half"
    return "Full"


def apply_hysteresis(prev, raw, ratio):
    if prev is None or raw == prev:
        return raw
    if prev == "Full":
        if raw == "Half"  and ratio < HALF_THRESH  + HYSTERESIS_MARGIN:
            return "Full"
        if raw == "Empty" and ratio < ALERT_THRESH + HYSTERESIS_MARGIN:
            return "Full"
    elif prev == "Half":
        if raw == "Full"  and ratio > HALF_THRESH  - HYSTERESIS_MARGIN:
            return "Half"
        if raw == "Empty" and ratio < ALERT_THRESH + HYSTERESIS_MARGIN:
            return "Half"
    elif prev == "Empty":
        if raw != "Empty" and ratio > ALERT_THRESH - HYSTERESIS_MARGIN:
            return "Empty"
    return raw




def save_debug(frame, tray_pts, tray_mask, empty_mask,
               ratio, smoothed, label, tid, fnum):
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.boundingRect(tray_pts)
    pad = 12
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)

    raw_crop = frame[y1:y2, x1:x2].copy()
    ann_crop = raw_crop.copy()

    pts_local = tray_pts - np.array([x1, y1])
    cv2.polylines(ann_crop, [pts_local], True, COLOURS[label], 2)

    overlap      = cv2.bitwise_and(tray_mask, empty_mask)
    overlap_crop = overlap[y1:y2, x1:x2]
    tray_crop    = tray_mask[y1:y2, x1:x2]

    if np.any(overlap_crop == 255):
        m = overlap_crop == 255
        ann_crop[m] = (ann_crop[m].astype(np.float32) * 0.4
                       + np.array([0, 140, 255], np.float32) * 0.6).astype(np.uint8)
    tray_only = (tray_crop == 255) & (overlap_crop != 255)
    if np.any(tray_only):
        ann_crop[tray_only] = (ann_crop[tray_only].astype(np.float32) * 0.85
                               + np.array([200, 50, 50], np.float32) * 0.15).astype(np.uint8)

    TH = 200
    def _r(img):
        if img.shape[0] == 0 or img.shape[1] == 0:
            return np.zeros((TH, TH, 3), np.uint8)
        s = TH / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1]*s)), TH))
    def _lbl(img, txt):
        out = np.zeros((28 + img.shape[0], img.shape[1], 3), np.uint8)
        out[28:] = img
        cv2.putText(out, txt, (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        return out

    left  = _lbl(_r(raw_crop), "RAW tray#%s" % tid)
    right = _lbl(_r(ann_crop),
                 "inst=%.3f smooth=%.3f -> %s" % (ratio, smoothed, label))
    mh = max(left.shape[0], right.shape[0])
    def _pad(img):
        if img.shape[0] < mh:
            return np.vstack([img, np.zeros((mh - img.shape[0], img.shape[1], 3), np.uint8)])
        return img
    cv2.imwrite(
        os.path.join(DEBUG_FOLDER, "f%04d_t%s_%s_%.3f.jpg" % (fnum, tid, label, ratio)),
        np.hstack([_pad(left), _pad(right)])
    )




def draw_tray_overlay(frame, pts, label, pct_empty, alert, tid):
    col = COLOURS[label]
    cv2.polylines(frame, [pts], True, col, 2)
    ov = frame.copy()
    cv2.fillPoly(ov, [pts], col)
    cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)

    if alert:
        x, y, bw, bh = cv2.boundingRect(pts)
        cv2.rectangle(frame, (x-3, y-3), (x+bw+3, y+bh+3), (0, 0, 255), 3)

    cx = int(np.mean(pts[:, 0]))
    cy = int(np.mean(pts[:, 1]))

    for text, dy, scale in [
        ("#%s  %s" % (tid, label),                                   -10, 0.55),
        ("%.0f%% empty / %.0f%% full" % (pct_empty, 100-pct_empty),  14, 0.48),
        ("REFILL NEEDED" if alert else "",                             34, 0.50),
    ]:
        if not text:
            continue
        is_alert = text == "REFILL NEEDED"
        tc = (0, 0, 255) if is_alert else (255, 255, 255)
        bc = (0, 0, 200) if is_alert else col
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        lx, ly = cx - tw//2, cy + dy
        cv2.rectangle(frame, (lx-4, ly-th-3), (lx+tw+4, ly+4), bc, -1)
        cv2.putText(frame, text, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, scale, tc, 2)


def draw_summary_panel(frame, records, fw, fh):
    if not records:
        return
    lh, pw = 26, 340
    ph = 44 + lh * len(records)
    x2, y1 = fw - 14, 14
    x1, y2 = max(0, x2-pw), min(fh-1, y1+ph)
    bg = frame[y1:y2, x1:x2]
    ov = bg.copy()
    cv2.rectangle(ov, (0,0), (ov.shape[1], ov.shape[0]), (20,20,20), -1)
    cv2.addWeighted(ov, 0.65, bg, 0.35, 0, bg)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (70,70,70), 1)
    cv2.putText(frame, "Tray Status", (x1+10, y1+26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 2)
    ty = y1 + 26 + lh
    for rec in sorted(records, key=lambda r: r["id"]):
        tag  = "  ALERT" if rec["alert"] else ""
        text = "#%s  %.0f%% empty / %.0f%% full  -> %s%s" % (
            rec["id"], rec["pct_empty"], 100-rec["pct_empty"], rec["label"], tag)
        cv2.putText(frame, text, (x1+10, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOURS[rec["label"]], 1)
        ty += lh


def draw_counter_panel(frame, counts, fh):
    px, py = 16, fh - 115
    for lbl in ("Full", "Half", "Empty"):
        cv2.putText(frame, "%s : %d" % (lbl, counts[lbl]), (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOURS[lbl], 2)
        py += 34



print("Loading models...")
tray_model  = YOLO(TRAY_MODEL_PATH)
empty_model = YOLO(EMPTY_MODEL_PATH)
print("Models loaded.\n")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open: %s" % VIDEO_PATH)

WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS    = cap.get(cv2.CAP_PROP_FPS) or 25.0
TOTAL  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"avc1")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))
if not writer.isOpened():
    out_avi = OUTPUT_VIDEO.replace(".mp4", ".avi")
    writer  = cv2.VideoWriter(out_avi, cv2.VideoWriter_fourcc(*"XVID"), FPS, (WIDTH, HEIGHT))
    print("avc1 unavailable - writing to %s" % out_avi)


ratio_history   = defaultdict(lambda: deque(maxlen=SMOOTH_WINDOW))
ema_state       = {}   
confirmed_label = {}
last_seen       = {}
cached_draw     = {}
cached_empty    = None

frame_count = 0
t_start     = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    do_debug  = frame_count <= DEBUG_FRAMES
    run_tray  = (frame_count == 1) or (frame_count % TRAY_SKIP  == 0)
    run_empty = (frame_count == 1) or (frame_count % EMPTY_SKIP == 0)

    out_frame = frame.copy()
    counts    = {"Full": 0, "Half": 0, "Empty": 0}
    records   = []

    
    if run_empty:
        empty_results = empty_model.predict(
            frame,
            conf=CONF_EMPTY,
            imgsz=IMG_SIZE_EMPTY,
            retina_masks=True,
            device=DEVICE,
            verbose=False,
        )[0]
        cached_empty = build_empty_mask(empty_results, HEIGHT, WIDTH)
    else:
        empty_results = None

    empty_mask = cached_empty if cached_empty is not None else np.zeros((HEIGHT, WIDTH), np.uint8)

    
    if run_tray:
        tray_results = tray_model.track(
            frame,
            persist=True,
            conf=CONF_TRAY,
            tracker=TRACKER_CFG,
            retina_masks=True,
            imgsz=IMG_SIZE_TRAY,
            device=DEVICE,
            verbose=False,
        )[0]

        track_ids = None
        if tray_results.boxes is not None and tray_results.boxes.id is not None:
            track_ids = tray_results.boxes.id.int().cpu().tolist()

        if tray_results.masks is not None:
            for i, seg in enumerate(tray_results.masks.xy):
                pts = np.round(seg).astype(np.int32)
                if pts.shape[0] < 3:
                    continue

                tid = track_ids[i] if (track_ids and i < len(track_ids)) else ("u%d" % i)

                ratio, tray_mask = compute_ratio(pts, empty_mask, HEIGHT, WIDTH)

                
                if tid not in ema_state:
                    ema_state[tid] = ratio
                else:
                    ema_state[tid] = EMA_ALPHA * ratio + (1 - EMA_ALPHA) * ema_state[tid]
                smoothed  = ema_state[tid]

                
                ratio_history[tid].append(ratio)
                last_seen[tid] = frame_count
                pct_empty = smoothed * 100.0

                raw   = classify(smoothed)
                label = apply_hysteresis(confirmed_label.get(tid), raw, smoothed)
                confirmed_label[tid] = label
                alert = smoothed >= ALERT_THRESH
                cached_draw[tid] = (pts, label, pct_empty, alert)

                if do_debug:
                    save_debug(frame, pts, tray_mask, empty_mask,
                               ratio, smoothed, label, tid, frame_count)

        
        for tid in [t for t, s in last_seen.items() if frame_count - s > STALE_AFTER]:
            ratio_history.pop(tid, None)
            ema_state.pop(tid, None)          
            confirmed_label.pop(tid, None)
            last_seen.pop(tid, None)
            cached_draw.pop(tid, None)

    
    for tid, (pts, label, pct_empty, alert) in cached_draw.items():
        draw_tray_overlay(out_frame, pts, label, pct_empty, alert, tid)
        counts[label] += 1
        records.append({"id": tid, "pct_empty": pct_empty, "label": label, "alert": alert})

    if empty_results is not None and empty_results.masks is not None:
        for seg in empty_results.masks.xy:
            cv2.polylines(out_frame, [np.round(seg).astype(np.int32)],
                          True, (0, 220, 255), 1)

    draw_summary_panel(out_frame, records, WIDTH, HEIGHT)
    draw_counter_panel(out_frame, counts, HEIGHT)
    cv2.putText(out_frame, "Frame %d" % frame_count,
                (WIDTH - 150, HEIGHT - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

    writer.write(out_frame)

    if frame_count % 50 == 0:
        elapsed    = time.time() - t_start
        fps_actual = frame_count / elapsed if elapsed > 0 else 0
        remaining  = ((TOTAL - frame_count) / fps_actual) if fps_actual > 0 else 0
        alerts     = sum(1 for r in records if r["alert"])
        print("  Frame %4d/%d | %.1f fps | ETA %dm%02ds | "
              "Full=%d Half=%d Empty=%d Alerts=%d" % (
              frame_count, TOTAL, fps_actual,
              int(remaining)//60, int(remaining)%60,
              counts["Full"], counts["Half"], counts["Empty"], alerts))

cap.release()
writer.release()
elapsed = time.time() - t_start
print("\nDone in %dm%02ds.  Output : %s" % (int(elapsed)//60, int(elapsed)%60, OUTPUT_VIDEO))
print("Debug crops   : %s  (first %d frames)" % (DEBUG_FOLDER, DEBUG_FRAMES))