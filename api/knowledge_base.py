import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define FAISS folder
FAISS_FOLDER = Path("faiss_index")  # Ensure this matches `pdf_to_faiss.py`

# Load FAISS index
def load_faiss_index():
    if not FAISS_FOLDER.exists():
        raise FileNotFoundError("❌ FAISS index folder not found. Run `pdf_to_faiss.py` first.")

    try:
        # Load embedding model
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Load FAISS index directly from folder (NO pickle)
        vector_store = FAISS.load_local(str(FAISS_FOLDER), embedding_model)

        print("✅ FAISS index loaded successfully!")
        return vector_store, embedding_model

    except Exception as e:
        raise RuntimeError(f"❌ Failed to load FAISS index: {str(e)}")

# Query FAISS index
def query_knowledge_base(query: str, top_k: int = 3) -> str:
    try:
        vector_store, embedding_model = load_faiss_index()
        
        # Search for relevant documents
        results = vector_store.similarity_search_with_score(query, k=top_k)

        if not results:
            return "I couldn't find relevant information in the knowledge base."

        # Format response
        response = "\n".join(f"- {doc[0].page_content} (Score: {doc[1]:.2f})" for doc in results)
        return f"📚 **Knowledge Base Results:**\n{response}"

    except Exception as e:
        return f"❌ Error querying FAISS index: {str(e)}"

if __name__ == "__main__":
    print(query_knowledge_base("What is AI?"))
