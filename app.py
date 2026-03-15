import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Load secrets: Streamlit Cloud uses st.secrets, local dev uses .env
load_dotenv()
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from document_processor import extract_text_from_pdf, extract_pages_from_pdf, get_document_chunks
from extractor import extract_lease_summary
from chat_engine import ChatEngine

st.set_page_config(page_title="AI Lease Chat", layout="wide")

st.title("📝 AI-Powered Lease Document Chat")
st.write("Upload a lease document (PDF) to automatically extract key terms and chat with the document.")

# Initialize session state for the chat engine and extraction state
if "chat_engine" not in st.session_state:
    st.session_state.chat_engine = ChatEngine()
if "summary" not in st.session_state:
    st.session_state.summary = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None and not st.session_state.document_processed:
    with st.spinner("Processing document..."):
        # 1. Read document
        # Load directly from BytesIO or save to tempfile if needed (PyPDF2 supports file-like objects)
        full_text = extract_text_from_pdf(uploaded_file)
        pages = extract_pages_from_pdf(uploaded_file)
        
        # 2. Extract structured summary
        st.session_state.summary = extract_lease_summary(full_text)
        
        # 3. Chunk and ingest into RAG
        chunks = get_document_chunks(pages)
        st.session_state.chat_engine.ingest_documents(chunks)
        
        st.session_state.document_processed = True
        st.success("Document processed successfully!")

# Layout: Split into two columns if document is processed
if st.session_state.document_processed:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📋 Extracted Lease Summary")
        if st.session_state.summary:
            summary = st.session_state.summary
            
            st.subheader("Parties & Premises")
            st.write(f"**Tenant:** {summary.tenant or 'N/A'}")
            st.write(f"**Landlord:** {summary.landlord or 'N/A'}")
            st.write(f"**DBA Name:** {summary.dba_name or 'N/A'}")
            st.write(f"**Address:** {summary.address or 'N/A'}")
            st.write(f"**Leased Area:** {summary.leased_area_sqft or 'N/A'}")
            st.write(f"**Permitted Use:** {summary.permitted_use or 'N/A'}")

            st.subheader("Key Dates & Financials")
            st.write(f"**Lease Start Date:** {summary.lease_start_date or 'N/A'}")
            st.write(f"**Lease End Date:** {summary.lease_end_date or 'N/A'}")
            st.write(f"**Rent Amount:** {summary.rent_amount or 'N/A'}")
            st.write(f"**Security Deposit:** {summary.security_deposit or 'N/A'}")
            
            st.subheader("Options & Clauses")
            if summary.renewal_options:
                for idx, opt in enumerate(summary.renewal_options):
                    st.write(f"**Renewal Option {idx+1}:** {opt.number_of_options} options, {opt.term_years} years. (Notice: {opt.notice_period})")
            else:
                st.write("No renewal options found.")
                
            if summary.termination_clauses:
                for idx, tc in enumerate(summary.termination_clauses):
                    st.write(f"**Termination Clause {idx+1}:** {tc.description}")
            else:
                st.write("No early termination clauses found.")
                
            st.subheader("Special Provisions")
            st.write(summary.special_provisions or 'N/A')
        else:
            st.error("Failed to extract summary.")

    with col2:
        st.header("💬 Chat with Document")
        
        # Display chat messages from history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.write(f"**Page {source['page']}:** {source['content_snippet']}")

        # Accept user input
        if prompt := st.chat_input("Ask a question about the lease..."):
            # Display user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chat_engine.ask(prompt)
                    answer = response["answer"]
                    sources = response["sources"]
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("View Sources"):
                            for source in sources:
                                st.write(f"**Page {source['page']}:** {source['content_snippet']}")
                    
            # Add assistant response to history
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
