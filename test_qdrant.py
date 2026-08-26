from qdrant_client import QdrantClient

# Connect to the local Qdrant server running in Docker
client = QdrantClient(host="localhost", port=6333)

# Print existing collections to verify connection
print("Connected to Qdrant!")
print("Collections:", client.get_collections())