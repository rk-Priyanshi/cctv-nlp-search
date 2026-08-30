from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import streamlit as st
client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)
COLLECTION_NAME = "cctv_frames"

# Create collection if it doesn't already exist
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=512,  # Matches standard CLIP model output size
            distance=Distance.COSINE
        ),
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully!")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")