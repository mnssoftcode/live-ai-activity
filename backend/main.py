from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ultralytics import YOLO
import io
import torch
import torchvision.transforms as T
from pytorchvideo.models.hub import slowfast_r50
import ffmpeg
import time

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model (person detection)
model = YOLO('yolov8n.pt')  # Use nano model for speed
PERSON_CLASS_ID = 0  # COCO class 0 is 'person'

# Load PyTorchVideo SlowFast model (pretrained on Kinetics-400)
action_model = slowfast_r50(pretrained=True)
action_model.eval()

# Kinetics-400 class names
KINETICS_CLASSES = [
    'abseiling', 'air drumming', 'answering questions', 'applauding', 'applying cream', 'archery', 'arm wrestling',
    # ... (full list: https://github.com/deepmind/kinetics-i3d/blob/master/data/label_map.txt)
    'yoga', 'zumba'
]

# For demo, use a subset (or load full list from file if needed)

# Simple tracker and buffer for each person
TRACKER_IOU_THRESH = 0.5
CLIP_LEN = 16  # frames per clip
person_buffers = {}  # id: {"frames": [np.ndarray], "box": [x1,y1,x2,y2], "last_seen": timestamp}
person_id_counter = 0

# Helper: IoU between two boxes
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# Helper: Preprocess clip for SlowFast
def preprocess_clip(frames):
    # Resize, center crop, convert to tensor, normalize
    transform = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    frames = [transform(f) for f in frames]
    clip = torch.stack(frames, dim=1)  # (C, T, H, W)
    # SlowFast expects two pathways: slow and fast
    fast_pathway = clip
    slow_indices = torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // 4).long()
    slow_pathway = clip[:, slow_indices, :, :]
    return [slow_pathway.unsqueeze(0), fast_pathway.unsqueeze(0)]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global person_id_counter
    # Read image from upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    # Run YOLOv8 detection
    results = model(img)[0]
    detected = []
    now = time.time()
    for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
        if int(cls) == PERSON_CLASS_ID:
            x1, y1, x2, y2 = map(int, box)
            detected.append({"box": [x1, y1, x2, y2], "conf": float(conf)})

    # Track persons by IoU
    updated_buffers = {}
    for det in detected:
        best_iou = 0
        best_id = None
        for pid, buf in person_buffers.items():
            i = iou(det["box"], buf["box"])
            if i > best_iou:
                best_iou = i
                best_id = pid
        if best_iou > TRACKER_IOU_THRESH:
            pid = best_id
        else:
            pid = person_id_counter
            person_id_counter += 1
        # Crop person
        crop = img[det["box"][1]:det["box"][3], det["box"][0]:det["box"][2]]
        if crop.size == 0:
            continue
        buf = person_buffers.get(pid, {"frames": [], "box": det["box"], "last_seen": now})
        buf["frames"].append(crop)
        buf["box"] = det["box"]
        buf["last_seen"] = now
        if len(buf["frames"]) > CLIP_LEN:
            buf["frames"] = buf["frames"][-CLIP_LEN:]
        updated_buffers[pid] = buf
    # Remove old tracks
    for pid, buf in person_buffers.items():
        if pid not in updated_buffers and now - buf["last_seen"] < 2:
            updated_buffers[pid] = buf
    person_buffers.clear()
    person_buffers.update(updated_buffers)

    # Prepare response
    boxes, labels, scores = [], [], []
    for pid, buf in person_buffers.items():
        box = buf["box"]
        frames = buf["frames"]
        if len(frames) == CLIP_LEN:
            try:
                clip = preprocess_clip(frames)
                with torch.no_grad():
                    preds = action_model(clip)
                    prob = torch.nn.functional.softmax(preds, dim=1)[0]
                    top_idx = prob.argmax().item()
                    label = KINETICS_CLASSES[top_idx] if top_idx < len(KINETICS_CLASSES) else 'unknown'
                    conf = float(prob[top_idx])
            except Exception as e:
                label = 'error'
                conf = 0.0
        else:
            label = 'detecting...'
            conf = 0.01
        boxes.append(box)
        labels.append(label)
        scores.append(conf)

    return JSONResponse({
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "count": len(boxes)
    }) 