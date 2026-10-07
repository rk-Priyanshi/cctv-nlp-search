import streamlit as st
import cv2
import os
import logging
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, PayloadSchemaType
from qdrant_client.http.exceptions import UnexpectedResponse

from embedder import get_text_embedding
from ingest_video import process_video_with_yolo
import drive_video

# Silence noisy/unrelated transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)


# -----------------------------
# Qdrant Connection
# -----------------------------

@st.cache_resource
def get_qdrant():
    return QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

client = get_qdrant()

COLLECTION_NAME = "cctv_frames"
def read_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

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
# Google Drive Video Selection
# -----------------------------

# Handle OAuth callback: Streamlit delivers the ?code= and ?state= query params here.
_qp = st.query_params
if "code" in _qp and not drive_video.is_authenticated():
    _returned_state = _qp.get("state")
    if not drive_video.verify_and_clear_state(_returned_state):
        # State missing or mismatched — possible CSRF; reject and clear URL.
        st.error(
            "⚠️ OAuth state mismatch. The sign-in attempt was rejected to prevent "
            "a potential CSRF attack. Please try signing in again."
        )
        st.query_params.clear()
    else:
        try:
            tokens = drive_video.exchange_code_for_tokens(_qp["code"])
            drive_video.save_tokens(tokens)
            # Clear the code/state from the URL to avoid re-exchange on reload
            st.query_params.clear()
            st.rerun()
        except Exception as _e:
            st.error(f"Google OAuth failed: {_e}")

with st.expander("📂 Select a video from Google Drive", expanded=False):

    if not drive_video.is_authenticated():
        auth_url = drive_video.get_auth_url()
        st.markdown(
            f"[🔑 Sign in with Google]({auth_url})",
            help="Opens Google sign-in. After authorising, you will be redirected back here.",
        )
    else:
        col_info, col_logout = st.columns([4, 1])
        with col_info:
            st.success("✅ Connected to Google Drive (read-only)")
        with col_logout:
            if st.button("Sign out", key="drive_signout"):
                drive_video.logout()
                st.rerun()

        # List videos in the authenticated account's Drive
        access_token = drive_video.get_access_token()
        if access_token:
            try:
                with st.spinner("Fetching video list from Google Drive…"):
                    drive_videos = drive_video.list_drive_videos(access_token)
            except Exception as _e:
                st.error(f"Could not list Drive videos: {_e}")
                drive_videos = []

            if not drive_videos:
                st.info("No video files found in your Google Drive.")
            else:
                video_options = {f["name"]: f for f in drive_videos}
                selected_name = st.selectbox(
                    "Choose a video:",
                    options=list(video_options.keys()),
                    key="drive_selected_video",
                )

                if st.button("⬇️ Download & Process from Drive", key="drive_process_btn"):
                    selected_file = video_options[selected_name]
                    try:
                        with st.spinner(f"Downloading '{selected_name}' from Google Drive…"):
                            local_path = drive_video.download_drive_video(
                                file_id=selected_file["id"],
                                filename=selected_name,
                                access_token=access_token,
                            )

                        # Store in session_state so the search section can find it
                        st.session_state["uploaded_video_path"] = local_path
                        st.session_state["uploaded_video_name"] = selected_name

                        with st.spinner("Processing video… This may take some time."):
                            process_video_with_yolo(local_path)

                        st.session_state["video_processed"] = True
                        st.success(
                            f"✅ '{selected_name}' processed successfully! You can now search it."
                        )
                    except Exception as _e:
                        st.error(f"Drive processing failed: {_e}")

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

    if uploaded_file is None and not st.session_state.get("uploaded_video_name"):

        st.warning("⚠️ Please upload a video first.")

    elif not st.session_state.get("video_processed", False):

        st.warning("⚠️ Please click 'Process Video' before searching.")

    elif not query_text:

        st.warning("⚠️ Please enter a search query.")

    else:

        with st.spinner("Searching video frames..."):

            # Convert text query into CLIP embedding
            t0 = time.perf_counter()
            query_vector = get_text_embedding(query_text)
            t_encode = (time.perf_counter() - t0) * 1000
            # Search ONLY inside the uploaded video
            try:
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
            except UnexpectedResponse as e:
                st.error(f"Qdrant error {e.status_code}: {e.content}")
                st.stop()

            results = response.points
            t_total = (time.perf_counter() - t0) * 1000
            st.caption(f"⚡ Text encoding: {t_encode:.0f} ms | Total search: {t_total:.0f} ms")

        # -----------------------------
        # Display Results
        # -----------------------------

        if not results:

            st.warning("No matching frames found in the uploaded video.")

        else:

            st.subheader(f"Top {len(results)} Matches for '{query_text}':")

            video_path = st.session_state["uploaded_video_path"]

            if not os.path.exists(video_path):
                st.error("Uploaded video file could not be found.")
                st.stop()

            # Load all matching frames in parallel
            with ThreadPoolExecutor(max_workers=len(results)) as pool:
                frames = list(pool.map(
                    lambda h: read_frame(video_path, h.payload.get("frame_number", 0)),
                    results
                ))

            cols = st.columns(len(results))

            for i, hit in enumerate(results):

                timestamp = hit.payload.get("timestamp", 0)
                frame_number = hit.payload.get("frame_number", 0)
                confidence = hit.score * 100
                frame_rgb = frames[i]

                with cols[i]:

                    st.markdown(f"**Match #{i + 1}**")

                    if frame_rgb is None:
                        st.error(f"Could not read frame {frame_number}.")
                    else:
                        bbox = hit.payload.get("bbox")
                        label = hit.payload.get("object_label", "")
                        if bbox and label != "full_frame":
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame_rgb, label, (x1, max(20, y1 - 8)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        st.image(frame_rgb, width="stretch")

                    st.write(f"⏱️ **Timestamp:** {timestamp:.2f}s")
                    st.write(f"🎯 **Confidence:** {confidence:.2f}%")
