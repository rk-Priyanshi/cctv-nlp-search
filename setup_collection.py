from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(
    url="https://81ee74e6-44fe-49f2-a7d7-4e5b2c4191b9.us-east-2-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MGQwMWM2N2QtODdlMy00Yjg2LWE3MjAtMjAzYzJkNWM0ZTBlIn0.6QqREzZi_shEIDoxmqBGS7xUc8AK-tpbgdDqZi1i0bI",
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