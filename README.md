# Gemini-Powered PDF Chatbot

A Streamlit-based chatbot that uses **Google's Gemini AI** to answer questions about PDF documents. Just upload a PDF, type your question, and get instant answers based on the document content.

---

## Features

- Extracts and processes PDF text
- Splits text into smart chunks using LangChain
- Embeds content using Gemini `embedding-001`
- Performs semantic search with FAISS
- Answers questions using Gemini 2.0 Flash model
- Clean and interactive Streamlit UI

  ## Requirements
streamlit ,
PyPDF2 ,
langchain ,
langchain-community ,
langchain-google-genai ,
faiss-cpu ,
google-generativeai ,

## Install these requirements using :-
pip install -r requirements.txt

Add your Google Gemini API key:

Open app.py and replace the line:
GEMINI_API_KEY = "YOUR_API_KEY_HERE"

▶️ How to Use :-
python -m streamlit run chatbot.py
