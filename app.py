import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline

# -------------------- UI CONFIG --------------------
st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("AI Study Assistant for Students")
st.write("Upload your notes and study smarter using AI")

# -------------------- FUNCTIONS --------------------

def extract_text_from_pdf(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def load_llm():
    generator = pipeline(
        "text-generation",
        model="google/flan-t5-base",
        max_length=512,
        temperature=0.3
    )
    llm = HuggingFacePipeline(pipeline=generator)
    return llm


def generate_summary(text, llm):
    prompt = f"""Summarize the following study material in simple student-friendly language:

{text[:2000]}

Summary:"""
    try:
        response = llm(prompt, max_length=512)
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                return response[0]['generated_text']
            elif isinstance(response[0], str):
                return response[0]
        elif isinstance(response, dict) and 'generated_text' in response:
            return response['generated_text']
        else:
            return str(response)
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def generate_mcqs(text, llm):
    prompt = f"""Based on this text, create exactly 10 multiple choice questions. For each question, provide:
Q1) [question text]
A) [option a]
B) [option b]
C) [option c]
D) [option d]
Answer: [correct letter]

Text: {text[:2000]}

Now create the questions:"""
    try:
        response = llm(prompt, max_length=1024)
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                return response[0]['generated_text']
            elif isinstance(response[0], str):
                return response[0]
        elif isinstance(response, dict) and 'generated_text' in response:
            return response['generated_text']
        else:
            return str(response)
    except Exception as e:
        return f"Error generating MCQs: {str(e)}"


def generate_flashcards(text, llm):
    prompt = f"""Create flashcards from the following content. Format as:
Card 1:
Q: [question]
A: [answer]

Card 2:
Q: [question]
A: [answer]

And so on...

Content: {text[:2000]}

Create flashcards:"""
    try:
        response = llm(prompt, max_length=512)
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                return response[0]['generated_text']
            elif isinstance(response[0], str):
                return response[0]
        elif isinstance(response, dict) and 'generated_text' in response:
            return response['generated_text']
        else:
            return str(response)
    except Exception as e:
        return f"Error generating flashcards: {str(e)}"


# -------------------- FILE UPLOAD --------------------

uploaded_file = st.file_uploader("Upload your study material (PDF only)", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        vectorstore = create_vector_store(raw_text)
        llm = load_llm()

    st.success("Document processed successfully!")

    # -------------------- TABS --------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Ask Questions",
        "Summary",
        "MCQs",
        "Flashcards"
    ])

    # -------------------- Q&A --------------------
    with tab1:
        st.subheader("Ask questions from your notes")
        query = st.text_input("Enter your question")

        if query:
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
                chain_type="stuff"
            )
 
            response = qa_chain.run(query)
            st.write("###Answer:")
            st.write(response)

    # -------------------- SUMMARY --------------------
    with tab2:
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                summary = generate_summary(raw_text, llm)
            st.write(summary)

    # -------------------- MCQs --------------------
    with tab3:
        if st.button("Generate MCQs"):
            with st.spinner("Generating MCQs..."):
                mcqs = generate_mcqs(raw_text, llm)
            st.write(mcqs)

    # -------------------- FLASHCARDS --------------------
    with tab4:
        if st.button("Generate Flashcards"):
            with st.spinner("Generating flashcards..."):
                flashcards = generate_flashcards(raw_text, llm)
            st.write(flashcards)

else:
    st.info("Upload a PDF to get started")
