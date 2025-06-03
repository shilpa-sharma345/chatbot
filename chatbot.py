import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# Gemini API Key
GEMINI_API_KEY = "Your GEMINI API KEY"

#  Streamlit App UI
st.header("📄 Gemini-Powered PDF Chatbot")

with st.sidebar:
    st.title("Upload a PDF")
    file = st.file_uploader("Upload a PDF file and ask questions", type="pdf")

if file is not None:
    # Read PDF content
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if text:
        #  Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n"],
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # 🔎 Embedding using Gemini (embedding-001 is stable and supported)
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )

        #  Create vector store
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)

        #  User question input
        user_question = st.text_input("Ask a question about the document:")

        if user_question:
            #  Find relevant chunks
            docs = vector_store.similarity_search(user_question)

            #  Use Gemini-2.0 Flash for QA
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=GEMINI_API_KEY,
                temperature=0.2,
            )

            #  QA Chain
            chain = load_qa_chain(llm, chain_type="stuff")
            result = chain.run(input_documents=docs, question=user_question)

            #  Show the answer
            st.write("### ✅ Answer:")
            st.write(result)
    else:
        st.warning("⚠️ Could not extract text from the PDF.")