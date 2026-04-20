import os
from dotenv import load_dotenv
from pinecone import Pinecone
from datasets import load_dataset

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

print("📥 Loading Wikipedia dataset from HuggingFace (streaming)...")
dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    split="train",
    streaming=True,
    trust_remote_code=True
)

from langchain_huggingface import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

batch_texts, batch_ids, batch_meta = [], [], []
count = 0
LIMIT = 50000

for item in dataset:
    text = item["text"][:1000]
    page_id = str(item["id"])
    url = f"https://en.wikipedia.org/wiki?curid={page_id}"

    batch_texts.append(text)
    batch_ids.append(page_id)
    batch_meta.append({
        "url": url,
        "title": item["title"],
        "text": text,
    })

    if len(batch_texts) == 100:
        vectors = embeddings_model.embed_documents(batch_texts)
        index.upsert(vectors=list(zip(batch_ids, vectors, batch_meta)))
        count += 100
        print(f"✅ {count} articles uploaded...")
        batch_texts, batch_ids, batch_meta = [], [], []

    if count >= LIMIT:
        break

# Upload remaining
if batch_texts:
    vectors = embeddings_model.embed_documents(batch_texts)
    index.upsert(vectors=list(zip(batch_ids, vectors, batch_meta)))
    count += len(batch_texts)

print(f"\n🎉 Done! {count} Wikipedia articles loaded into Pinecone.")