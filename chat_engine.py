import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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
        self._build_chain()

    def _build_chain(self):
        """
        Builds the conversational retrieval chain.
        """
        system_prompt = (
            "You are an expert legal assistant. Use the following pieces of retrieved context to answer the question.\n"
            "Always include clear source citations pointing to the Page Number (e.g., 'Extracted from Page X') based on the document provided in the context.\n"
            "If you don't know the answer or the context doesn't contain it, just say that you don't know.\n"
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, question_answer_chain)

    def ask(self, query: str):
        """
        Asks a question and returns the answer with sources.
        """
        if not self.qa_chain:
            return {"answer": "Please upload a document first.", "sources": []}
            
        result = self.qa_chain.invoke({"input": query})
        
        answer = result["answer"]
        source_docs = result["context"]
        
        sources = []
        for doc in source_docs:
            sources.append({
                "page": doc.metadata.get("page", "Unknown"),
                "content_snippet": doc.page_content[:200] + "..."
            })
            
        return {"answer": answer, "sources": sources}
