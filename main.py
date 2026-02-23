from fastapi import FastAPI
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
import os

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rag")
def rag_query(query: str):
    # Exemplo simples - ajuste pro seu
    llm = Ollama(model="jurema:7b")  # Aponta pro jurema-llm
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    qdrant = Qdrant.from_existing_collection(
        embedding=embeddings,
        url="http://qdrant:6333",
        collection_name="tributario"
    )
    retriever = qdrant.as_retriever()
    # ... resto da chain aqui
    return {"result": "exemplo"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
