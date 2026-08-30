import cv2
import os
from ultralytics import YOLO
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from embedder import get_image_embedding
import streamlit as st
client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)
COLLECTION_NAME = "cctv_frames"

# Load lightweight YOLOv8 nano model
yolo_model = YOLO("yolov8n.pt")

def process_video_with_yolo(video_path, sample_rate_sec=1):
    """
    Extracts frames, uses YOLOv8 to crop detected objects,
    embeds cropped regions via CLIP, and saves them to Qdrant.
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Could not read video FPS.")
        return

    frame_interval = int(fps * sample_rate_sec)
    frame_count = 0
    point_id = 0
    points = []

    os.makedirs("temp_crops", exist_ok=True)
    print(f"Processing video with YOLO: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps
            
            # Run YOLO detection on the frame
            results = yolo_model(frame, verbose=False)[0]
            
            for box in results.boxes:
                # Extract box coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                conf = float(box.conf[0])

                # Filter out low-confidence detections
                if conf < 0.4:
                    continue

                # Crop object region from frame
                cropped_img = frame[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue

                crop_path = f"temp_crops/crop_{point_id}.jpg"
                cv2.imwrite(crop_path, cropped_img)

                # Generate vector embedding for cropped object
                vector = get_image_embedding(crop_path)

                # Create Qdrant record with bounding box payload metadata
                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "timestamp": timestamp_sec,
                        "frame_number": frame_count,
                        "video_source": os.path.basename(video_path),
                        "object_label": label,
                        "bbox": [x1, y1, x2, y2]
                    }
                )
                points.append(point)
                print(f"Detected '{label}' at {timestamp_sec:.2f}s (Point ID: {point_id})")
                point_id += 1

        frame_count += 1

    cap.release()

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"\nSuccessfully stored {len(points)} object embeddings in Qdrant!")

if __name__ == "__main__":
    process_video_with_yolo("Sample1.mp4", sample_rate_sec=1)