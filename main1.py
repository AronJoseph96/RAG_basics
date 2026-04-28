import os
from dotenv import load_dotenv
from google import genai

# Import your custom modules
from pdf_processor import load_and_split_pdf
from database import setup_vector_db, get_embedding
from llm_engine import ask_gemini

def main():
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "..", "AronJoseph-Resume.pdf")
    db_path = os.path.join(base_dir, "chroma_db")

    # 1. Process
    print("Processing Resume...")
    chunks = load_and_split_pdf(pdf_path)
    
    # 2. Store
    collection = setup_vector_db(client, chunks, db_path)

    # 3. Interactive Loop
    while True:
        query = input("\nAsk about the candidate (or 'exit'): ")
        if query.lower() == 'exit': break

        # Retrieval
        query_vec = get_embedding(client, query)
        results = collection.query(query_embeddings=[query_vec], n_results=5)
        context = "\n".join(results["documents"][0])

        # Generation
        answer = ask_gemini(client, context, query)
        print(f"\n[ASSISTANT]: {answer}")

if __name__ == "__main__":
    main()