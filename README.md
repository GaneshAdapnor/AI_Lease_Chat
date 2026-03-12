# AI-Powered Lease Chat System

This project is an end-to-end AI document analysis pipeline that ingests a legal lease document (PDF), extracts key structured data fields matching the assignment constraints, and provides a Retrieval-Augmented Generation (RAG) chat interface with source citations. 

## 🚀 Setup Instructions

1. **Install Dependencies:**
   Ensure Python 3.10+ is installed. Run the command:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   You must have an `OPENAI_API_KEY` set. Modify the `.env` file to contain your API Key:
   ```env
   OPENAI_API_KEY=your_key_here
   ```

3. **Run the App:**
   Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## 🏗 Pipeline Overview

1. **Ingestion (`document_processor.py`):**
   We use `PyPDF2` to read the document. Text is extracted in two ways:
   - Full string format for structured extraction.
   - Page-mapped format chunked by `RecursiveCharacterTextSplitter` (keeping page numbers as metadata) for RAG.

2. **Extraction (`extractor.py`):**
   Utilizes **LangChain's Structured Outputs** bound to a `Pydantic` schema (`LeaseSummary`). The full text is sent to `gpt-4o-mini` with a system prompt instructing it to extract the required lease fields (Tenant, Landlord, Rent, Renewal, Dates, etc.).

3. **Retrieval & Chat (`chat_engine.py`):**
   Creates embeddings using `OpenAIEmbeddings` and stores chunks in an in-memory `FAISS` vector database. When a user asks a question, a Conversational Retrieval Chain finds the Top-4 relevant chunks and passes them to `gpt-4o-mini`, explicitly instructed to cite the source page.

4. **UI (`app.py`):**
   A standard Streamlit interface showing the parsed fields on the left and the interactive chat on the right.

---

## 🔎 Edge Case Analysis

Here are at least 5 potential issues the system may encounter, and how they are addressed or could be mitigated:

1. **Ambiguous or missing fields in the lease:**
   - *Issue*: The script might attempt to hallucinate values when a lease lacks a specific field (e.g., DBA name or Guarantor is missing).
   - *Strategy*: Prompts are instructed to return `null` or empty if a field is ambiguous. The Pydantic schema uses `Optional` for every field, preventing failure, and the UI gracefully renders 'N/A' for nulls.

2. **Conflicting clauses / extremely long sections:**
   - *Issue*: A lease might have amendments or duplicate sections. A long section might exceed chunk sizes, losing context.
   - *Strategy*: By sending the *entire* document to the 128k context window of `gpt-4o-mini` for the static extraction, the model resolves conflicts holistically rather than relying on RAG for extraction. For chat, semantic search with FAISS brings in multiple chunks (k=4) so the LLM sees various perspectives.

3. **Queries with multiple possible answers:**
   - *Issue*: Asking \"What is the term?\" might refer to the Initial Term, Renewal Term, or Sublease Term.
   - *Strategy*: The system prompt in `chat_engine.py` enforces citing source pages. If an answer covers multiple terms, the LLM will display different pages and sections, making the distinction clear to the user.

4. **Scanned / OCR-dependent PDFs:**
   - *Issue*: `PyPDF2` does not inherently perform OCR. If a scanned, flat-image PDF is uploaded, text extraction will fail or yield garbage.
   - *Strategy*: Currently, the app expects text-encoded PDFs. A mitigation for production would be integrating an OCR library like `pytesseract` or `Unstructured` block parsing fallback if `PyPDF2` returns an empty string.

5. **Hallucination of Legal Advice:**
   - *Issue*: The chat system might accidentally provide proactive legal advice rather than simply summarizing the existing document text.
   - *Strategy*: The system prompt explicitly roles the LLM as an \"expert legal assistant\" tasked *only* with using the *retrieved context* to answer questions. It explicitly contains instructions to say \"I don't know\" if the text isn't relevant, minimizing arbitrary legal advice.

## Discussion of Limitations and Trade-offs
- **In-memory Vector Store:** FAISS is rebuilt every time the application is refreshed. While fast and suitable for single-document pipelines, scaling to multi-tenant architectures requires persistent vector databases (e.g., Pinecone or ChromaDB).
- **Cost vs Context Size:** By feeding the entire 30+ pages to `gpt-4o-mini` for extraction, accuracy is high but token cost scales linearly. A cost mitigation would be doing map-reduce summaries over chunks first, but at the potential risk of dropping specific facts.
