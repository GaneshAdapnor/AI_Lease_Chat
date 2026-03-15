import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

class ChatEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def ingest_documents(self, chunks):
        """
        Ingests the chunked documents into a FAISS vector store.
        """
        documents = [
            Document(page_content=chunk["text"], metadata=chunk["metadata"])
            for chunk in chunks
        ]
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
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
