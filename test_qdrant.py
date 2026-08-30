from qdrant_client import QdrantClient

# Connect to the local Qdrant server running in Docker
client = QdrantClient(
    url="https://81ee74e6-44fe-49f2-a7d7-4e5b2c4191b9.us-east-2-0.aws.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6MGQwMWM2N2QtODdlMy00Yjg2LWE3MjAtMjAzYzJkNWM0ZTBlIn0.6QqREzZi_shEIDoxmqBGS7xUc8AK-tpbgdDqZi1i0bI",
)

# Print existing collections to verify connection
print("Connected to Qdrant!")
print("Collections:", client.get_collections())