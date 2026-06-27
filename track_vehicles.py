# vehicle_track_filtered.py
import cv2
import torch
import numpy as np
from collections import deque
import os, sys

# -------------------------
# CONFIG - change if needed
# -------------------------
VIDEO_PATH = "source_video_feed.avi"
MODEL_PATH = "yolov5/runs/train/exp/weights/best.pt"
CONF_THRES = 0.20      # keep moderate; lower to show more raw detections
NMS_IOU = 0.45

FPS_OVERRIDE = None

# matching/tracking
IOU_MATCH_THRESHOLD = 0.25
MAX_MATCH_DISTANCE = 80
SPEED_HISTORY_LEN = 5
ACC_HISTORY_LEN = 4

# behavior thresholds
ACCEL_THRESH = 1.5
BRAKE_THRESH = -1.5

# meters-per-pixel heuristic
AVG_VEH_LEN_M = 4.5

# filtering rules (tuned to cut those vertical strips)
MAX_BOX_AREA_FRAC = 0.30    # discard boxes covering >30% of frame area
MAX_ASPECT_RATIO = 3.0      # discard boxes where h / w > 3.0 (too tall)
MIN_ASPECT_RATIO = 0.33     # discard boxes where h / w < 0.33 (too flat)

OUTPUT_VIDEO = "output_detected.avi"

# -------------------------
# helpers
# -------------------------
def sanitize_bbox(bbox, W, H):
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(W-1, round(x1))))
    y1 = int(max(0, min(H-1, round(y1))))
    x2 = int(max(0, min(W-1, round(x2))))
    y2 = int(max(0, min(H-1, round(y2))))
    if x2 <= x1: x2 = min(W-1, x1+2)
    if y2 <= y1: y2 = min(H-1, y1+2)
    return (x1, y1, x2, y2)

def xyxy_to_xywh(b):
    x1,y1,x2,y2 = b
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = int(x1 + w/2)
    cy = int(y1 + h/2)
    return cx, cy, w, h

def iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    area_a = max(0,(ax2-ax1)) * max(0,(ay2-ay1))
    area_b = max(0,(bx2-bx1)) * max(0,(by2-by1))
    union = area_a + area_b - inter
    return inter/union if union>0 else 0.0

def smooth(dq):
    return float(sum(dq) / len(dq)) if dq else 0.0

# Simple greedy NMS using IoU, keeps boxes with higher confidence
def simple_nms(boxes, scores, iou_thresh=0.6):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(scores)[::-1]  # high->low
    keep = []
    boxes = np.array(boxes)
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        idxs = rest[ious <= iou_thresh]
    return keep

# -------------------------
# Load YOLO model
# -------------------------
print("Loading YOLO model:", MODEL_PATH)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, verbose=False)
model.conf = CONF_THRES
model.iou = NMS_IOU
print("Model loaded. conf:", model.conf, "iou:", model.iou)

# -------------------------
# Open video + writer
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: cannot open video:", VIDEO_PATH)
    sys.exit(1)

cap_fps = cap.get(cv2.CAP_PROP_FPS) or 0
fps = FPS_OVERRIDE or cap_fps or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {W}x{H} @ {fps} FPS")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (W,H))

# Try start window thread
try:
    cv2.startWindowThread()
except Exception:
    pass
WINDOW_NAME = "Vehicle Tracking - filtered bboxes"

# -------------------------
# Tracking state
# -------------------------
tracks = {}
next_id = 0
frame_idx = 0

# -------------------------
# Main loop
# -------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    t = frame_idx / fps

    results = model(frame)
    raw = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class]

    raw_count = len(raw)
    # build sanitized list and apply heuristic filters
    candidates = []
    for r in raw:
        x1,y1,x2,y2,conf,cls = r
        if conf < CONF_THRES:
            continue
        bbox = sanitize_bbox((x1,y1,x2,y2), W, H)
        cx,cy,wbox,hbox = xyxy_to_xywh(bbox)
        area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
        area_frac = area / (W*H)
        aspect = float(hbox) / float(wbox) if wbox>0 else 999
        # filter bad shapes/areas
        if area_frac > MAX_BOX_AREA_FRAC:
            continue
        if aspect > MAX_ASPECT_RATIO or aspect < MIN_ASPECT_RATIO:
            continue
        candidates.append({'bbox':bbox,'centroid':(cx,cy),'w':wbox,'h':hbox,'conf':float(conf)})

    # apply simple NMS to cleaned candidates (to remove duplicates)
    boxes_for_nms = [c['bbox'] for c in candidates]
    scores_for_nms = [c['conf'] for c in candidates]
    keep_idx = simple_nms(boxes_for_nms, scores_for_nms, iou_thresh=0.6)
    filtered = [candidates[i] for i in keep_idx]

    # debug prints: how many raw vs filtered
    if frame_idx % 10 == 1:
        print(f"[frame {frame_idx}] raw={raw_count} cand={len(candidates)} final={len(filtered)}")

    # Now tracking: IoU first, centroid fallback
    used_det_idx = set()
    used_track_ids = set()

    # IoU matching
    if len(tracks) > 0 and len(filtered) > 0:
        track_ids = list(tracks.keys())
        iou_mat = np.zeros((len(track_ids), len(filtered)), dtype=float)
        for i, tid in enumerate(track_ids):
            tbbox = sanitize_bbox(tracks[tid]['bbox'], W, H)
            for j, det in enumerate(filtered):
                iou_mat[i,j] = iou(tbbox, det['bbox'])
        for i, tid in enumerate(track_ids):
            best_j = int(np.argmax(iou_mat[i]))
            best_iou = float(iou_mat[i, best_j])
            if best_iou >= IOU_MATCH_THRESHOLD and best_j not in used_det_idx and tid not in used_track_ids:
                det = filtered[best_j]
                tracks[tid]['bbox'] = det['bbox']
                tracks[tid]['centroids'].append(det['centroid'])
                tracks[tid]['times'].append(t)
                tracks[tid]['last_seen'] = t
                used_det_idx.add(best_j)
                used_track_ids.add(tid)
                # speed & acc
                if len(tracks[tid]['centroids']) >= 2:
                    x_prev,y_prev = tracks[tid]['centroids'][-2]
                    x_cur,y_cur = tracks[tid]['centroids'][-1]
                    pixel_disp = np.hypot(x_cur - x_prev, y_cur - y_prev)
                    dt = tracks[tid]['times'][-1] - tracks[tid]['times'][-2] or (1.0/fps)
                    meters_per_pixel = AVG_VEH_LEN_M / max(1, det['w'])
                    meters_disp = pixel_disp * meters_per_pixel
                    speed_m_s = meters_disp / dt
                    tracks[tid]['speed_hist'].append(speed_m_s * 3.6)
                    if len(tracks[tid]['speed_hist']) >= 2:
                        v2 = tracks[tid]['speed_hist'][-1] / 3.6
                        v1 = tracks[tid]['speed_hist'][-2] / 3.6
                        acc = (v2 - v1) / dt
                        tracks[tid]['acc_hist'].append(acc)

    # centroid fallback for unmatched tracks
    unmatched_tracks = [tid for tid in tracks.keys() if tid not in used_track_ids]
    unmatched_dets = [i for i in range(len(filtered)) if i not in used_det_idx]

    for tid in unmatched_tracks:
        best_j = None; best_dist = None
        tx, ty = tracks[tid]['centroids'][-1]
        for j in unmatched_dets:
            dx = filtered[j]['centroid'][0] - tx
            dy = filtered[j]['centroid'][1] - ty
            dist = np.hypot(dx, dy)
            if best_dist is None or dist < best_dist:
                best_dist = dist; best_j = j
        if best_j is not None and best_dist is not None and best_dist < MAX_MATCH_DISTANCE:
            det = filtered[best_j]
            tracks[tid]['bbox'] = det['bbox']
            tracks[tid]['centroids'].append(det['centroid'])
            tracks[tid]['times'].append(t)
            tracks[tid]['last_seen'] = t
            used_det_idx.add(best_j)
            if best_j in unmatched_dets:
                unmatched_dets.remove(best_j)
            if len(tracks[tid]['centroids']) >= 2:
                x_prev,y_prev = tracks[tid]['centroids'][-2]
                x_cur,y_cur = tracks[tid]['centroids'][-1]
                pixel_disp = np.hypot(x_cur - x_prev, y_cur - y_prev)
                dt = tracks[tid]['times'][-1] - tracks[tid]['times'][-2] or (1.0/fps)
                meters_per_pixel = AVG_VEH_LEN_M / max(1, det['w'])
                meters_disp = pixel_disp * meters_per_pixel
                speed_m_s = meters_disp / dt
                tracks[tid]['speed_hist'].append(speed_m_s * 3.6)
                if len(tracks[tid]['speed_hist']) >= 2:
                    v2 = tracks[tid]['speed_hist'][-1] / 3.6
                    v1 = tracks[tid]['speed_hist'][-2] / 3.6
                    acc = (v2 - v1) / dt
                    tracks[tid]['acc_hist'].append(acc)

    # create new tracks for remaining detections
    for j, det in enumerate(filtered):
        if j in used_det_idx:
            continue
        tracks[next_id] = {
            'bbox': det['bbox'],
            'centroids': [det['centroid']],
            'times': [t],
            'speed_hist': deque(maxlen=SPEED_HISTORY_LEN),
            'acc_hist': deque(maxlen=ACC_HISTORY_LEN),
            'last_seen': t
        }
        tracks[next_id]['speed_hist'].append(0.0)
        next_id += 1

    # remove stale tracks
    stale = [tid for tid, obj in tracks.items() if t - obj['last_seen'] > 1.5]
    for sid in stale:
        del tracks[sid]

    # draw each track with small green box and label
    for tid, obj in tracks.items():
        bbox = sanitize_bbox(obj['bbox'], W, H)
        x1,y1,x2,y2 = bbox
        cx,cy = obj['centroids'][-1]

        speed_kmph = smooth(obj['speed_hist'])
        acc_val = obj['acc_hist'][-1] if len(obj['acc_hist'])>0 else 0.0

        # simple lane-change heuristic (lateral moves)
        lane_change = False
        if len(obj['centroids']) >= 3:
            x_prev2 = obj['centroids'][-3][0]
            x_prev1 = obj['centroids'][-2][0]
            x_cur = obj['centroids'][-1][0]
            lateral1 = abs(x_prev1 - x_prev2)
            lateral2 = abs(x_cur - x_prev1)
            if lateral1 > 15 and lateral2 > 15 and abs(x_cur - x_prev2) > 25:
                lane_change = True

        if acc_val > ACCEL_THRESH:
            behavior = "Accelerating"
        elif acc_val < BRAKE_THRESH:
            behavior = "Braking"
        elif lane_change:
            behavior = "Lane Changing"
        else:
            behavior = "Normal"

        label = f"ID:{tid} | vehicle | speed={speed_kmph:.1f} kmph | {behavior}"

        # draw bbox and label anchored on top-left of bbox
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 3, (0,0,255), -1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx1 = x1
        ly2 = y1 - 4
        if ly2 - th < 0:
            ly2 = y1 + th + 8
        lx2 = min(W-1, lx1 + tw + 6)
        ly1 = max(0, ly2 - th - 6)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0,0,0), -1)
        cv2.putText(frame, label, (lx1 + 3, ly2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

    # small on-screen hint
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    out.write(frame)
    try:
        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception:
        # headless: just keep writing
        pass

# cleanup
cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except Exception:
    pass

print("Finished. Output saved to:", os.path.abspath(OUTPUT_VIDEO))
