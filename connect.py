from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DB_FAISS_PATH = "vectorstores/db_faiss"

# 1) Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2) Load FAISS
db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

# 3) Retriever (increase k)
retriever = db.as_retriever(search_kwargs={"k": 8})

# 4) LLM (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

# 5) Prompt (strict but usable)
prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not present in the context, say: I don't know.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def print_sources(docs):
    print("\nSOURCES USED:")
    seen = set()
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        key = (src, page)
        if key not in seen:
            print(f"- {src} (page {page})")
            seen.add(key)

def print_chunks(docs, max_chars=600):
    print("\n--- RETRIEVED CHUNKS (DEBUG) ---")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        print(f"\n[{i}] {src} (page {page})")
        print(d.page_content[:max_chars])

# ==============================
# MAIN
# ==============================

query = input("Write Query Here: ").strip()

# Retrieve documents
docs = retriever.invoke(query)

# Debug: show what was retrieved
print_chunks(docs)

context = format_docs(docs)

# Generate answer
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"context": context, "question": query})

print("\nRESULT:\n", result)

# Print sources
print_sources(docs)
