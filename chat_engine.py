import os
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

BATCH_SIZE = 20      # chunks per embedding API call
BATCH_DELAY = 1.0    # seconds between batches

class ChatEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None
        self.retriever = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def ingest_documents(self, chunks):
        """
        Ingests chunked documents into a FAISS vector store in small batches
        to avoid hitting OpenAI rate limits.
        """
        documents = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in chunks
        ]

        # Process first batch to create the vector store
        self.vector_store = FAISS.from_documents(documents[:BATCH_SIZE], self.embeddings)

        # Add remaining batches with a small delay between each
        for i in range(BATCH_SIZE, len(documents), BATCH_SIZE):
            time.sleep(BATCH_DELAY)
            batch = documents[i:i + BATCH_SIZE]
            self.vector_store.add_documents(batch)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

    def ask(self, query: str):
        """
        Asks a question manually constructing the prompt and returns the answer with sources.
        """
        if not self.retriever:
            return {"answer": "Please upload a document first.", "sources": []}
            
        source_docs = self.retriever.invoke(query)
        context_text = "\n\n".join([f"Page {doc.metadata.get('page', 'Unknown')}: {doc.page_content}" for doc in source_docs])
        
        system_prompt = (
            "You are an expert legal assistant. Use the following pieces of retrieved context to answer the question.\n"
            "Always include clear source citations pointing to the Page Number (e.g., 'Extracted from Page X') based on the document provided in the context.\n"
            "If you don't know the answer or the context doesn't contain it, just say that you don't know.\n"
            f"Context: {context_text}"
        )
        
        messages = [
            ("system", system_prompt),
            ("human", query)
        ]
        
        result = self.llm.invoke(messages)
        answer = result.content
        
        sources = []
        for doc in source_docs:
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "content_snippet": doc.page_content[:200] + "..."
            })
            
        return {"answer": answer, "sources": sources}
