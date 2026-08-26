# 📹 Smart CCTV Natural Language Search

An AI-powered video search application that allows users to search inside CCTV footage using natural text prompts.

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Vector Database:** Qdrant (running via Docker)
* **Embedding Model:** OpenAI CLIP (`openai/clip-vit-base-patch32` via Hugging Face Transformers)
* **Frontend:** Streamlit
* **Video Processing:** OpenCV & Pillow

## 🚀 Getting Started

### 1. Prerequisites
Ensure Docker is installed and running, then start the Qdrant container:
```bash
docker run -d -p 6333:6333 -p 6344:6344 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant