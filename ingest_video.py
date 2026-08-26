import cv2
import os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from embedder import get_image_embedding

# Initialize Qdrant Client
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "cctv_frames"

def process_video(video_path, sample_rate_sec=1):
    """
    Extracts frames from a video every `sample_rate_sec` seconds,
    generates embeddings, and stores them in Qdrant.
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
    saved_count = 0
    points = []

    # Temporary folder to save frames for CLIP processing
    os.makedirs("temp_frames", exist_ok=True)

    print(f"Processing video: {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame at specified second intervals
        if frame_count % frame_interval == 0:
            timestamp_sec = frame_count / fps
            temp_image_path = f"temp_frames/frame_{saved_count}.jpg"
            
            # Save frame temporarily
            cv2.imwrite(temp_image_path, frame)
            
            # Generate embedding vector using embedder.py
            vector = get_image_embedding(temp_image_path)
            
            # Create a point object for Qdrant
            point = PointStruct(
                id=saved_count,
                vector=vector,
                payload={
                    "timestamp": timestamp_sec,
                    "frame_number": frame_count,
                    "video_source": os.path.basename(video_path)
                }
            )
            points.append(point)
            saved_count += 1
            print(f"Processed frame at {timestamp_sec:.2f}s (Point ID: {saved_count - 1})")

        frame_count += 1

    cap.release()

    # Upload all extracted vector points to Qdrant
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"\nSuccessfully stored {len(points)} video frames in Qdrant!")

if __name__ == "__main__":
    # Place a sample mp4 video in your folder and update the filename here:
    process_video("Sample1.mp4", sample_rate_sec=1)