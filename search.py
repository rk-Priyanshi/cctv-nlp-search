from qdrant_client import QdrantClient
from embedder import get_text_embedding

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "cctv_frames"

def search_video(query_text, top_k=3):
    """Searches stored video frames using natural text prompts."""
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
    
    # 3. Print match results
    print(f"\nTop {top_k} matching timestamps for '{query_text}':")
    print("-" * 50)
    for i, hit in enumerate(results, 1):
        timestamp = hit.payload.get("timestamp", 0)
        confidence = hit.score * 100
        video_name = hit.payload.get("video_source", "Unknown")
        
        print(f"Match #{i}:")
        print(f"  - Timestamp  : {timestamp:.2f} seconds")
        print(f"  - Confidence : {confidence:.2f}%")
        print(f"  - Video File : {video_name}")
        print("-" * 50)

if __name__ == "__main__":
    # Test natural text queries against your CCTV footage:
    search_video("Where is the watch at the end?", top_k=3)