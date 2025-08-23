import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("GOOGLE_API_KEY not found in .env file!")
    st.stop()

genai.configure(api_key=api_key)

# -----------------------------
# 2. Build Wikipedia Knowledge Base
# -----------------------------
@st.cache_resource
def load_knowledge_base(topic: str):
    loader = WikipediaLoader(query=topic, load_max_docs=3)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )
    return qa_chain, llm


# -----------------------------
# 3. Generate Interview Questions
# -----------------------------
def generate_interview_questions(topic: str, llm, num_questions: int = 10):
    prompt = f"""
    You are an interviewer preparing to ask questions about the topic: {topic}.
    Generate {num_questions} thoughtful, varied, and human-like interview questions
    that cover fundamental, technical, and future aspects of this topic.
    """
    response = llm.invoke(prompt)
    # Ensure it's a clean list
    questions = [q.strip("- ").strip() for q in response.content.split("\n") if q.strip()]
    return questions[:num_questions]


# -----------------------------
# 4. Streamlit Interface
# -----------------------------
st.title("🎙️ AgentCast Interview Simulator")
st.write("Enter a topic and let the AI simulate an interview with 10 assumed questions.")

topic = st.text_input("Enter a topic (e.g., Quantum Computing):")

if topic:
    with st.spinner(f"Loading knowledge base for '{topic}'..."):
        qa_chain, llm = load_knowledge_base(topic)

    with st.spinner("Generating interview questions..."):
        questions = generate_interview_questions(topic, llm)

    st.success("Interview questions generated!")
    st.subheader("🤔 Predicted Interview Questions")
    for i, q in enumerate(questions, 1):
        st.write(f"**Q{i}. {q}**")

    st.subheader("🎤 Simulated Interview Session")
    for i, q in enumerate(questions, 1):
        st.markdown(f"**Q{i}. {q}**")
        answer = qa_chain.invoke({"query": q})
        st.markdown(f"**A{i}.** {answer['result']}")
        st.markdown("---")
