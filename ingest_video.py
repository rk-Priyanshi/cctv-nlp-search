import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLOWorld
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from embedder import get_image_embedding_from_pil

COLLECTION_NAME = "cctv_frames"

# ---------------- Tunable settings ----------------
CONF_THRESHOLD = 0.30        # lower to 0.3-0.4 if YOLO-World misses things
IOU_THRESHOLD = 0.45         # NMS overlap threshold
MIN_BOX_SIZE = 30           # discard boxes smaller than 80x80 px
MAX_BOX_AREA_RATIO = 0.6    # discard "loose" boxes covering >60% of the frame
BOX_PADDING = 0.12          # 8% context padding around each box
CLIP_INPUT_SIZE = 224
UPSERT_BATCH = 64
STORE_FULL_FRAME = True     # also embed the whole frame for scene-level queries

# Broad open vocabulary: add anything your users might search for.
VOCAB = [
    "person", "face", "man", "thief", "child", "girl",
    "backpack", "handbag", "suitcase", "bag", "box", "package", "umbrella",
    "helmet", "hat", "cap", "glasses", "mask", "jacket", "shirt", "shoe",
    "car", "truck", "bus", "van", "motorcycle", "bicycle", "scooter",
    "license plate", "bottle", "cup", "phone", "laptop", "camera", "key",
    "knife", "tool", "weapon", "dog", "cat", "bird",
    "chair", "table", "door", "bench", "sign", "trash can", "cart", "luggage",
    # kitchen / food
    "egg", "plate", "jar", "measuring cup", "tap", "mixer", "green onions",
    "stove", "strainer", "pan", "spoon", "whisk", "basin", "chopstick",
    "noodles", "food",
    # home / scene
    "plants", "window", "calendar", "frames", "books", "bed", "pillows",
    # nature / effects
    "water", "hills", "clouds", "steam", "potted plant"
]

# ---------------- Lazy initialisation (no work at import time) ----------------
_client = None
_model = None


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
        )
    return _client


def get_model():
    global _model
    if _model is None:
        _model = YOLOWorld("yolov8m-worldv2.pt")
        _model.set_classes(VOCAB)
    return _model


def ensure_collection():
    client = get_client()
    names = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )

    # Index so we can filter results by video name
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="video_source",
            field_schema="keyword",
        )
    except Exception as e:
        print(f"Payload index note: {e}")


# ---------------- Crop helpers ----------------
def letterbox_square(img_bgr, size=CLIP_INPUT_SIZE):
    """Resize to fit inside size x size and pad to a square, so CLIP's
    center-crop doesn't cut off the edges of non-square crops."""
    h, w = img_bgr.shape[:2]
    scale = size / max(h, w)
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    resized = cv2.resize(
        img_bgr, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp
    )
    canvas = np.full((size, size, 3), 127, dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))


def valid_box(x1, y1, x2, y2, frame_w, frame_h):
    w, h = x2 - x1, y2 - y1
    if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
        return False
    if (w * h) / float(frame_w * frame_h) > MAX_BOX_AREA_RATIO:
        return False
    return True


def pad_box(x1, y1, x2, y2, frame_w, frame_h, pad=BOX_PADDING):
    pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return (max(0, x1 - pw), max(0, y1 - ph),
            min(frame_w, x2 + pw), min(frame_h, y2 + ph))


# ---------------- Main pipeline ----------------
def process_video_with_yolo(video_path, sample_rate_sec=0.25):
    """Extracts frames, detects objects with YOLO-World, embeds clean crops
    (and the full frame) with CLIP, and stores them in Qdrant."""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    ensure_collection()
    client = get_client()
    model = get_model()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not read video FPS.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(fps * sample_rate_sec))
    frame_count = 0
    point_id = client.count(collection_name=COLLECTION_NAME, exact=True).count
    points, total = [], 0
    video_name = os.path.basename(video_path)

    def add_point(vector, ts, fno, label, bbox, conf):
        nonlocal point_id, total
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "timestamp": ts,
                "frame_number": fno,
                "video_source": video_name,
                "video_path": video_path,
                "object_label": label,
                "confidence": conf,
                "bbox": bbox,
            },
        ))
        point_id += 1
        total += 1

    def flush():
        nonlocal points
        if points:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

    print(f"Processing video with YOLO-World: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            ts = frame_count / fps

            results = model.predict(
                frame,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                agnostic_nms=True,
                verbose=False,
            )[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                label = results.names[int(box.cls[0])]

                if not valid_box(x1, y1, x2, y2, frame_w, frame_h):
                    continue

                px1, py1, px2, py2 = pad_box(x1, y1, x2, y2, frame_w, frame_h)
                crop = frame[py1:py2, px1:px2]
                if crop.size == 0:
                    continue
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                vec = get_image_embedding_from_pil(letterbox_square(crop))
                add_point(vec, ts, frame_count, label, [x1, y1, x2, y2], conf)

            if STORE_FULL_FRAME:
                vec = get_image_embedding_from_pil(letterbox_square(frame))
                add_point(vec, ts, frame_count, "full_frame", [0, 0, frame_w, frame_h], 1.0)

            if len(points) >= UPSERT_BATCH:
                flush()

        frame_count += 1

    cap.release()
    flush()
    print(f"\nSuccessfully stored {total} embeddings in Qdrant!")


if __name__ == "__main__":
    process_video_with_yolo("Sample1.mp4", sample_rate_sec=0.10)
