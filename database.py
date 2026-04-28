import chromadb

def get_embedding(client, text):
    response = client.models.embed_content(
        model = "models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def setup_vector_db(client, chunks, db_path):
    chroma_client = chromadb.PersistentClient(path=db_path)
    collection = chroma_client.get_or_create_collection('resume_vault')

    if collection.count() == 0:
        print('indexing new resume data...')
        for i, chunk in enumerate(chunks):
            vector = get_embedding(client, chunk)
            collection.add(ids=[str(i)], embeddings=[vector], documents=[chunk])
    
    # MOVE THIS LINE HERE (out of the if block)
    return collection
