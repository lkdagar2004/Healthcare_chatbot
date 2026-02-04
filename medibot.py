import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

DB_FAISS_PATH = "vectorstores/db_faiss"

# ----------------------------
# Cache FAISS + Embeddings
# ----------------------------
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY")
    )

# ----------------------------
# Prompt
# ----------------------------
prompt_template = ChatPromptTemplate.from_template("""
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

def format_sources(docs):
    sources = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "unknown")
        sources.append(f"{src} (page {page})")
    # remove duplicates
    return list(dict.fromkeys(sources))

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Healthcare RAG Bot", layout="centered")
    st.title("Healthcare Chatbot (RAG + Groq)")

    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found. Add it in your .env file.")
        st.stop()

    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 6})
    llm = load_llm()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    user_query = st.chat_input("Ask something from your PDFs...")

    if user_query:
        # show user message
        st.chat_message("user").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # retrieve docs
        docs = retriever.invoke(user_query)
        context = format_docs(docs)

        # generate answer
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": user_query})

        # show assistant message
        st.chat_message("assistant").markdown(answer)

        # show sources (expandable)
        with st.expander("Sources"):
            for s in format_sources(docs):
                st.write(s)

        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
