import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import platform

# Set page configuration
st.set_page_config(page_title='Generación Aumentada por Recuperación (RAG)', layout='wide')

# Title and Image
st.title('Generación Aumentada por Recuperación (RAG) 💬')
image = Image.open('Chat_pdf.png')
st.image(image, width=350)

# Display Python version
st.write("Versión de Python:", platform.python_version())

# Sidebar for input key
st.sidebar.header("Bienvenido!")
st.sidebar.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado.")
ke = st.sidebar.text_input('Ingresa tu Clave', type='password')  # Use password input for security

# Set OpenAI API key
if ke:
    os.environ['OPENAI_API_KEY'] = ke

# File uploader for PDF
st.subheader("Carga tu archivo PDF para análisis")
pdf = st.file_uploader("Selecciona un archivo PDF", type="pdf")

# Process PDF and generate response
if pdf is not None:
    # Read and extract text from the PDF
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ''  # Avoid errors if text extraction fails

    # Split text into chunks for processing
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)

    # Create embeddings from text chunks
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # User input for question
    st.subheader("¿Qué quieres saber sobre el documento?")
    user_question = st.text_area("Escribe tu pregunta aquí:")

    # Generate response based on the user question
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-4o-mini")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
        
        # Display the response
        st.write("### Respuesta:")
        st.success(response)  # Use success message style for better visibility
else:
    st.warning("Por favor, carga un archivo PDF para empezar el análisis.")
