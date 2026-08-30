from qdrant_client import QdrantClient

import streamlit as st
client = QdrantClient(
    url=st.secrets["QDRANT_URL"],
    api_key=st.secrets["QDRANT_API_KEY"]
)

# Print existing collections to verify connection
print("Connected to Qdrant!")
print("Collections:", client.get_collections())