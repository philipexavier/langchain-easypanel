from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

app = FastAPI()

class Query(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "LangChain 1.2 OK"}

@app.post("/rag")
def rag_query(q: Query):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://jurema-llm:11434")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    
    llm = Ollama(model="jurema:7b", base_url=base_url)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=base_url)
    
    qdrant = Qdrant.from_existing_collection(
        embedding=embeddings,
        url=qdrant_url,
        collection_name="tributario"  # Seu índice de normas
    )
    
    retriever = qdrant.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_template(
        "Você é consultor tributário. Baseado nestes docs da reforma tributária:\n{context}\n\nPergunta: {query}\nResponda com regras aplicáveis e citações."
    )
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({
        "context": retriever.get_relevant_documents(q.query),
        "query": q.query
    })
    
    return {"result": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
