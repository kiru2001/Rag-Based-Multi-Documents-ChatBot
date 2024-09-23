import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
load_dotenv()

 


groq_api_key = os.getenv('GROQ_API_KEY')
google_api_key = os.getenv("GOOGLE_API_KEY")

def load_document(file):
    name, extension = os.path.splitext(file.name)
    
    with NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    if extension == '.pdf':
        st.write(f'Loading {file.name}')
        loader = PyPDFLoader(temp_file_path)
    elif extension == '.docx':
        st.write(f'Loading {file.name}')
        loader = Docx2txtLoader(temp_file_path)
    elif extension == '.txt':
        loader = TextLoader(temp_file_path)
    else:
        st.write('Document format is not supported!')
        return None

    data = loader.load()
    os.remove(temp_file_path)
    return data

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_llm=llm
    )
    return conversation_chain

def handle_userinput():
    # Get the current user question from session state
    user_question = st.session_state.user_question
    
    # Handle the user's input and generate a response
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

    # Clear the input field by resetting session state
    st.session_state.user_question = ""

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    st.header("BURI BURI ZAIMON :pig:")

    # Text input with on_change function
    st.text_input(
        "Ask a question about your documents:",
        value=st.session_state.user_question,
        key="user_question",
        on_change=handle_userinput  # Trigger the function on change
    )

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader('Choose a file', type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    # Load and concatenate text from all uploaded files
                    raw_text = ""
                    for pdf in pdf_docs:
                        text = load_document(pdf)
                        if text:
                            raw_text += " ".join([page.page_content for page in text])
                    
                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()


 