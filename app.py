import streamlit as st
import cv2
import os
import logging
from qdrant_client import QdrantClient
from embedder import get_text_embedding

# Silence noisy/unrelated transformers docstring-validation warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Setup Qdrant connection
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "cctv_frames"

# Folder where your source videos actually live (update if needed)
VIDEO_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="CCTV Video Search", layout="wide")
st.title("📹 Smart CCTV Natural Language Search")

st.markdown("Search inside your CCTV video using plain text prompts!")

# Search Bar Input
query_text = st.text_input("Enter search query:", placeholder="e.g., a person walking, red car, empty room")
top_k = st.slider("Number of results to display:", min_value=1, max_value=5, value=3)

if st.button("Search Video") and query_text:
    with st.spinner("Searching video frames..."):
        # Convert text query into CLIP embedding
        query_vector = get_text_embedding(query_text)

        # Query Qdrant using updated API
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        results = response.points

    if not results:
        st.warning("No matching frames found.")
    else:
        st.subheader(f"Top {len(results)} Matches for '{query_text}':")
        cols = st.columns(len(results))

        for i, hit in enumerate(results):
            timestamp = hit.payload.get("timestamp", 0)
            confidence = hit.score * 100
            video_name = hit.payload.get("video_source", "Unknown")
            frame_number = hit.payload.get("frame_number", 0)

            video_path = os.path.join(VIDEO_DIR, video_name)

            with cols[i]:
                st.markdown(f"**Match #{i+1}**")

                if not os.path.exists(video_path):
                    st.error(f"Video file not found: {video_path}")
                else:
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error(f"Could not open video: {video_path}")
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        ret, frame = cap.read()
                        cap.release()

                        if ret:
                            # Convert BGR (OpenCV) to RGB (Streamlit display)
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            st.image(frame_rgb, use_container_width=True)
                        else:
                            st.error(f"Could not read frame {frame_number} from {video_name}")

                st.write(f"⏱️ **Timestamp:** {timestamp:.2f}s")
                st.write(f"🎯 **Confidence:** {confidence:.2f}%")