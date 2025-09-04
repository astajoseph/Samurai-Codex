import asyncio
import nest_asyncio
import streamlit as st
import os
import time
import json
import uuid
import re
from collections import Counter
import itertools
import pandas as pd

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Async fix for Streamlit
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

# 🔑 API Keys
GROQ_API_KEY = "........................"
GOOGLE_API_KEY = "......................"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# 🧠 LLM Setup
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="gemma2-9b-it")

# 📌 Prompts
qa_prompt = ChatPromptTemplate.from_template("""
You are a legal document assistant. 
- Summarize content in simple language. 
- Identify potential risks. 
- Explain complex clauses. 
Answer only from the provided context. 
<context>
{context}
<context>
Question: {input}
""")

summary_prompt = ChatPromptTemplate.from_template("""
Summarize this legal document in simple, everyday language:Make
Cover all the important, complex,tricky legal terms and clauses while summarizing. 
Explain all the  legal clause in simple terms and highlight any risks mentioned in the document.                                            
<context>
{context}
<context>
""")

clause_prompt = ChatPromptTemplate.from_template("""
Explain the following legal clause in simple terms and highlight any risks:
{input}
""")

# Custom clause patterns for historical contracts
CLAUSE_PATTERNS = [
     (r'\bparties\b|\bbetween\b', 'PARTIES'),
    (r'\bobject\b|\bpurpose\b', 'OBJECT_PURPOSE'),
    (r'\bconsideration\b|\bpayment\b|\bfees?\b|\brent\b', 'CONSIDERATION_PAYMENT'),
    (r'\bterm\b|\bduration\b|\bvalidity\b', 'TERM_DURATION'),
    (r'\bobligations?\b|\bduties\b|\bresponsibilit(y|ies)\b', 'OBLIGATIONS'),
    (r'\bright(s)?\b|\bprivileges?\b', 'RIGHTS'),
    (r'\btermination\b|\bcancellation\b|\brescind\b', 'TERMINATION'),
    (r'\bliabilit(y|ies)\b|\bindemnif(y|ication)\b', 'LIABILITY_INDEMNITY'),
    (r'\bconfidential(ity)?\b|\bsecrecy\b', 'CONFIDENTIALITY'),
    (r'\bdispute\b|\barbitration\b|\bjurisdiction\b', 'DISPUTE_RESOLUTION'),
    (r'\bgoverning\s+law\b|\bapplicable\s+law\b', 'GOVERNING_LAW'),
    (r'\bforce\s+majeure\b|\bact\s+of\s+god\b', 'FORCE_MAJEURE'),
    (r'\bnotice\b|\bcommunication\b', 'NOTICE_COMMUNICATION'),
    (r'\bsign(ed|ature)?\b|\bexecution\b', 'SIGNATURE_EXECUTION')
]

def identify_clauses(text):
    """Identify legal clauses using regex patterns"""
    clauses = []
    for pattern, label in CLAUSE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            clauses.append(label)
    return clauses

# Helper to identify source of each chunk
class SourceDoc:
    def __init__(self, page_content, source):
        self.page_content = page_content
        self.source = source
        self.metadata = {"source": source}
        self.id = str(uuid.uuid4())

# 🖼️ UI
st.set_page_config(page_title="Legal Document Assistant", layout="wide")
st.title("📚 Legal Document Q&A Assistant")
st.markdown("Empowering users to understand legal documents with AI.")

# 📤 Upload PDF
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])
if uploaded_file:
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    os.makedirs("uploaded_docs", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Document uploaded successfully!")

# 📊 Sidebar Status
with st.sidebar:
    st.header("🛠️ Status")
    if "vectors" in st.session_state:
        st.success("Vector DB Ready")
    else:
        st.warning("Vector DB Not Initialized")

rag_path = "civil_law(RAG).json"
if os.path.exists(rag_path):
    with open(rag_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    civil_rag = data.get("civil_law_data", [])
else:
    civil_rag = []

# 📌 Embedding Function
def vector_embedding(path):
    from concurrent.futures import ThreadPoolExecutor
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)

    def process_pdf_chunk(doc):
        found_clauses = identify_clauses(doc.page_content)
        metadata = {"source": "PDF", "clauses": found_clauses}
        pdf_doc = SourceDoc(doc.page_content, "PDF")
        pdf_doc.metadata.update(metadata)
        return pdf_doc

    def process_rag_entry(entry):
        if entry.get("type") == "clause":
            text = entry["clause"] + "\n" + entry["layman_terms"]
        elif entry.get("type") == "term":
            text = entry["term"] + "\n" + entry["layman_example"]
        else:
            return None
        found_clauses = identify_clauses(text)
        metadata = {"source": "RAG", "clauses": found_clauses}
        rag_doc = SourceDoc(text, "RAG")
        rag_doc.metadata.update(metadata)
        return rag_doc

    with ThreadPoolExecutor() as executor:
        pdf_docs = list(executor.map(process_pdf_chunk, final_docs))
        rag_docs = list(executor.map(process_rag_entry, civil_rag))
        rag_docs = [doc for doc in rag_docs if doc is not None]

    all_docs = pdf_docs + rag_docs
    st.session_state.vectors = FAISS.from_documents(all_docs, st.session_state.embeddings)
    st.session_state.final_docs = all_docs

# 📎 Embed Button
if uploaded_file and st.button("📥 Embed Document"):
    vector_embedding(file_path)
    st.success("✅ Vector Store DB is Ready")

# 📝 Summarize Document
if uploaded_file and st.button("📝 Summarize Document"):
    if "final_docs" in st.session_state:
        summary_chain = create_stuff_documents_chain(llm, summary_prompt)
        response = summary_chain.invoke({'context': st.session_state.final_docs})
        st.subheader("📄 Document Summary")
        # Handle both dict and string responses
        if isinstance(response, dict):
            st.write(response.get('answer', response))
        else:
            st.write(response)


# ❓ Q&A
prompt1 = st.text_input("Ask a question about the document")
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, qa_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("⏱️ Response time:", round(time.process_time() - start, 2), "seconds")

    st.subheader("💡 Answer")
    st.write(response['answer'])

    with st.expander("📑 Relevant Document Chunks"):
        for doc in response["context"]:
            # If doc is a SourceDoc, show its source
            source = getattr(doc, 'source', 'Unknown')
            st.write(f"Source: {source}")
            st.write(doc.page_content)
            st.write("---")

# Clause Co-occurrence Analysis
def get_all_detected_clauses(docs):
    all_clauses = []
    for doc in docs:
        clauses = doc.metadata.get("clauses", [])
        if clauses:
            all_clauses.append(clauses)
    return all_clauses

def show_cooccurrence_matrix(docs):
    all_clauses = get_all_detected_clauses(docs)
    pairs = []
    for clauses in all_clauses:
        pairs.extend(itertools.combinations(sorted(set(clauses)), 2))
    co_occurrence = Counter(pairs)
    if co_occurrence:
        df = pd.DataFrame(
            {(a,b): count for (a,b), count in co_occurrence.items()},
            index=[0]
        ).T.sort_values(0, ascending=False)
        st.subheader("Clause Co-Occurrence Matrix")
        st.dataframe(df)
    else:
        st.info("No clause pairs detected for co-occurrence analysis.")

# Advice logic for missing and unusual clauses
EXPECTED_CLAUSES = ['TERM_DURATION', 'RENT_PAYMENT', 'SECURITY_DEPOSIT', 'TERMINATION']

def advice_on_missing_clauses(detected_clauses, expected_clauses=EXPECTED_CLAUSES):
    missing = [c for c in expected_clauses if c not in detected_clauses]
    if missing:
        return f"⚠️ Missing standard clauses: {', '.join(missing)}. Consider adding them for better protection."
    return "✅ All standard clauses are present."

def advice_on_unusual_combinations(detected_clauses):
    if 'TERMINATION' in detected_clauses and 'NOTICE_PERIOD' not in detected_clauses:
        return "⚠️ TERMINATION clause exists but no NOTICE PERIOD. This may weaken protection for one party."
    return ""

# Load risk analysis CSV
risk_df = pd.read_csv("legal_contract_clauses.csv")

# Load lawyer RAG
with open("lawyer(RAG).json", "r", encoding="utf-8") as f:
    lawyer_rag = json.load(f)

def get_clause_risk(clause_type):
    match = risk_df[risk_df['clause_type'].str.lower() == clause_type.lower()]
    if not match.empty:
        return match['risk_level'].iloc[0]
    return "unknown"

def get_top_lawyers(area="Civil", top_n=3):
    filtered = [l for l in lawyer_rag if l.get("specialization", "").lower() == area.lower()]
    filtered.sort(key=lambda x: -x.get("experience", 0))
    return filtered[:top_n]

# After embedding, add this to your UI:
if "final_docs" in st.session_state:
    show_cooccurrence_matrix(st.session_state.final_docs)
    all_detected = set()
    for doc in st.session_state.final_docs:
        all_detected.update(doc.metadata.get("clauses", []))
    advice1 = advice_on_missing_clauses(all_detected)
    advice2 = advice_on_unusual_combinations(all_detected)
    st.subheader("🛡️ Contract Advice")
    st.write(advice1)
    if advice2:
        st.write(advice2)
    # Risk logic: if any clause is high risk, suggest lawyers and stop further Q&A
    risky = [c for c in all_detected if get_clause_risk(c) == "high"]
    if risky:
        st.subheader("⚠️ High Risk Clauses Detected")
        st.write(f"Clauses: {', '.join(risky)}")
        top_lawyers = get_top_lawyers(area="Civil", top_n=3)
        st.subheader("👨‍⚖️ Top Civil Lawyers Recommended")
        for lawyer in top_lawyers:
            st.write(f"{lawyer['name']} ({lawyer['location']}, {lawyer['experience']} yrs)")
        st.stop()

# ⚠️ Disclaimer
st.markdown("""
---
⚠️ *Disclaimer: This tool provides simplified explanations of legal documents. It is not a substitute for professional legal advice.*
""")
