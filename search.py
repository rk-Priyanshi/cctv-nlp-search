from qdrant_client import QdrantClient
from embedder import get_text_embedding

import streamlit as st
client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)
COLLECTION_NAME = "cctv_frames"

def search_video(query_text, top_k=3):
    """Searches stored object crops using natural text prompts."""
    print(f"\nSearching for: '{query_text}'...")
    
    # 1. Convert text prompt into a vector using CLIP
    query_vector = get_text_embedding(query_text)
    
    # 2. Perform vector search in Qdrant (new API: query_points instead of search)
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k
    )
    results = response.points
    
    if not results:
        print("No matching objects found.")
        return

    # 3. Print match results
    print(f"\nTop {len(results)} matching objects for '{query_text}':")
    print("-" * 50)
    for i, hit in enumerate(results, 1):
        timestamp = hit.payload.get("timestamp", 0)
        confidence = hit.score * 100
        video_name = hit.payload.get("video_source", "Unknown")
        object_label = hit.payload.get("object_label", "unknown")
        bbox = hit.payload.get("bbox", None)
        frame_number = hit.payload.get("frame_number", 0)
        
        print(f"Match #{i}:")
        print(f"  - Object     : {object_label}")
        print(f"  - Timestamp  : {timestamp:.2f} seconds")
        print(f"  - Confidence : {confidence:.2f}%")
        print(f"  - Video File : {video_name}")
        print(f"  - Frame #    : {frame_number}")
        if bbox:
            print(f"  - BBox (x1,y1,x2,y2) : {bbox}")
        print("-" * 50)

    return results

if __name__ == "__main__":
    # Test natural text queries against your CCTV footage:
    search_video("Where is the watch at the end?", top_k=3)