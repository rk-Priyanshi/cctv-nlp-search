from functools import lru_cache
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

def get_image_embeddings_batch(images, batch_size=32):
    """Embeds a list of PIL images in batches (much faster than one by one)."""
    vectors = []
    for i in range(0, len(images), batch_size):
        chunk = [im.convert("RGB") for im in images[i:i + batch_size]]
        inputs = processor(images=chunk, return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        if torch.is_tensor(outputs):
            feats = outputs
        elif hasattr(outputs, "image_embeds"):
            feats = outputs.image_embeds
        elif hasattr(outputs, "pooler_output"):
            feats = outputs.pooler_output
        else:
            raise TypeError(f"Unexpected output type from get_image_features: {type(outputs)}")

        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        vectors.extend(feats.tolist())
    return vectors


def get_image_embedding_from_pil(image):
    """Single-image version, kept for compatibility."""
    return get_image_embeddings_batch([image])[0]


def get_image_embedding(image_path):
    """Kept for backward compatibility: embeds an image file."""
    return get_image_embedding_from_pil(Image.open(image_path))

@lru_cache(maxsize=256)
def get_text_embedding(text_query):
    """Converts a text query string into a vector list."""
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)

        if torch.is_tensor(outputs):
            text_features = outputs
        elif hasattr(outputs, "text_embeds"):
            text_features = outputs.text_embeds
        elif hasattr(outputs, "pooler_output"):
            text_features = outputs.pooler_output
        else:
            raise TypeError(f"Unexpected output type from get_text_features: {type(outputs)}")

    # Normalize vector to unit length
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.squeeze().tolist()
