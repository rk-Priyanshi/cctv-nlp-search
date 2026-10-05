import streamlit as st
import cv2
import os
import logging
import tempfile

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from embedder import get_text_embedding
from ingest_video import process_video_with_yolo

# Silence noisy/unrelated transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


# -----------------------------
# Qdrant Connection
# -----------------------------

client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

COLLECTION_NAME = "cctv_frames"


# -----------------------------
# Streamlit Page
# -----------------------------

st.set_page_config(
    page_title="CCTV Video Search",
    layout="wide"
)

st.title("📹 Smart CCTV Natural Language Search")

st.markdown(
    "Upload your CCTV video, process it, and search it using plain text prompts!"
)


# -----------------------------
# Upload Video
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your CCTV video",
    type=["mp4", "avi", "mov", "mkv"]
)


if uploaded_file is not None:

    video_path = os.path.join(
        tempfile.gettempdir(),
        uploaded_file.name
    )

    # Save uploaded video temporarily
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Store video information
    st.session_state["uploaded_video_path"] = video_path
    st.session_state["uploaded_video_name"] = uploaded_file.name

    st.success(f"Video uploaded: {uploaded_file.name}")

    # -----------------------------
    # Process Video Button
    # -----------------------------

    if st.button("⚙️ Process Video"):

        with st.spinner(
            "Processing video... This may take some time."
        ):
            process_video_with_yolo(video_path)

        st.session_state["video_processed"] = True

        st.success(
            "✅ Video processed successfully! You can now search it."
        )


# -----------------------------
# Search Section
# -----------------------------

query_text = st.text_input(
    "Enter search query:",
    placeholder="e.g., a person walking, red car, empty room"
)

top_k = st.slider(
    "Number of results to display:",
    min_value=1,
    max_value=5,
    value=3
)


# -----------------------------
# Search Button
# -----------------------------

if st.button("🔍 Search Video"):

    if uploaded_file is None:

        st.warning("⚠️ Please upload a video first.")

    elif not st.session_state.get("video_processed", False):

        st.warning("⚠️ Please click 'Process Video' before searching.")

    elif not query_text:

        st.warning("⚠️ Please enter a search query.")

    else:

        with st.spinner("Searching video frames..."):

            # Convert text query into CLIP embedding
            query_vector = get_text_embedding(query_text)

            # Search ONLY inside the uploaded video
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="video_source",
                            match=MatchValue(
                                value=st.session_state["uploaded_video_name"]
                            )
                        )
                    ]
                ),
                limit=top_k
            )

            results = response.points


        # -----------------------------
        # Display Results
        # -----------------------------

        if not results:

            st.warning(
                "No matching frames found in the uploaded video."
            )

        else:

            st.subheader(
                f"Top {len(results)} Matches for '{query_text}':"
            )

            cols = st.columns(len(results))

            for i, hit in enumerate(results):

                timestamp = hit.payload.get(
                    "timestamp",
                    0
                )

                confidence = hit.score * 100

                frame_number = hit.payload.get(
                    "frame_number",
                    0
                )

                # Use the currently uploaded video
                video_path = st.session_state[
                    "uploaded_video_path"
                ]

                with cols[i]:

                    st.markdown(
                        f"**Match #{i + 1}**"
                    )

                    if not os.path.exists(video_path):

                        st.error(
                            "Uploaded video file could not be found."
                        )

                    else:

                        cap = cv2.VideoCapture(video_path)

                        if not cap.isOpened():

                            st.error(
                                "Could not open uploaded video."
                            )

                        else:

                            # Move to the matching frame
                            cap.set(
                                cv2.CAP_PROP_POS_FRAMES,
                                frame_number
                            )

                            ret, frame = cap.read()

                            cap.release()

                            if ret:
                                                                # Convert BGR → RGB
                                frame_rgb = cv2.cvtColor(
                                    frame,
                                    cv2.COLOR_BGR2RGB
                                )

                                # Draw the matched object's bounding box
                                bbox = hit.payload.get("bbox")
                                label = hit.payload.get("object_label", "")

                                if bbox and label != "full_frame":
                                    x1, y1, x2, y2 = bbox
                                    cv2.rectangle(
                                        frame_rgb,
                                        (x1, y1),
                                        (x2, y2),
                                        (0, 255, 0),
                                        3
                                    )
                                    cv2.putText(
                                        frame_rgb,
                                        label,
                                        (x1, max(20, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.8,
                                        (0, 255, 0),
                                        2
                                    )

                                st.image(
                                    frame_rgb,
                                    use_container_width=True
                                )

                            else:

                                st.error(
                                    f"Could not read frame {frame_number}."
                                )

                    st.write(
                        f"⏱️ **Timestamp:** {timestamp:.2f}s"
                    )

                    st.write(
                        f"🎯 **Confidence:** {confidence:.2f}%"
                    )
