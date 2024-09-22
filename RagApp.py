from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec  # Updated Pinecone initialization
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as LangchainPinecone  # Updated import
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate 
import google.generativeai as genai

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

PINECONE_INDEX = "rag-app-index"


if PINECONE_INDEX not in pc.list_indexes().names():
   
    embedding_dimension = 768  
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=embedding_dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',region='us-east-1'       
        )
    )
# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text: 
                text += extracted_text
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and upload vectors to Pinecone
def get_vector_store(text_chunks,namespace):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = LangchainPinecone.from_texts(
        text_chunks,
         embeddings,
          index_name=PINECONE_INDEX,
          namespace=namespace)
    # vector_store.save_local("faiss_index")

# Function to create the conversational QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the questions as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." 
    Don't provide the wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and perform QA
def user_input(user_question,namespace):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Connect to the existing Pinecone index
    vector_store = LangchainPinecone.from_existing_index(
        PINECONE_INDEX, 
        embeddings,
        namespace=namespace
        )
    # Perform similarity search
    docs = vector_store.similarity_search(user_question)
    # Get the QA chain
    chain = get_conversational_chain()
    # Get the response
    response = chain(
        {"input_documents": docs,
         "question": user_question},
        return_only_outputs=True
    )
    print(response["output_text"])
    st.write("Response: ",response["output_text"])

import streamlit as st
from PyPDF2 import PdfReader

# Function to extract text from PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text

import streamlit as st



def main():
    st.set_page_config("Rag Bot Using Gemini-API and Pinecone-DB")
    st.header("Chat With Your Data")

    # Initializing session state variables
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'pdf_docs' not in st.session_state:
        st.session_state.pdf_docs = []

    with st.sidebar:
        st.title("Sidebar")

        # Allow multiple PDFs to be uploaded
        uploaded_files = st.file_uploader("Upload your documents as PDF files and Click Submit 2 Process", type="pdf", accept_multiple_files=True)

        if st.button("Submit 2 Process"):
            if uploaded_files:
                with st.spinner("Processing"):
                    # Extract and process the text for all uploaded PDFs
                    for pdf in uploaded_files:
                        current_pdf_name = pdf.name
                        st.write(f"Processing: {current_pdf_name}")
                        
                        # Extract text from the current PDF
                        raw_text = get_pdf_text([pdf])
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks, current_pdf_name)

                    st.success("Processing complete!")
                    st.session_state.pdf_processed = True
                    st.session_state.pdf_docs = uploaded_files
            else:
                st.warning("No PDFs were uploaded. Please upload your files.")

    if st.session_state.pdf_processed:
        selected_pdf = st.selectbox("Select a PDF file that you query:", [pdf.name for pdf in st.session_state.pdf_docs])
        #selected_pdf = st.radio("Select a PDF file that you query:", [pdf.name for pdf in st.session_state.pdf_docs])
        
        user_question = st.text_input("Ask a question regarding your data:")
        
        if user_question:
            user_input(user_question, selected_pdf)
    else:
        st.info("Please upload and process PDFs before you can query them.")

if __name__ == "__main__":
    main()
    
    
    
