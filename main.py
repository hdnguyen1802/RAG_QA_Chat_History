from history_aware_retriever import *
from question_answer_chain import *

import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings,GoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains import create_retrieval_chain

# --- Page Configuration ---
st.set_page_config(page_title="RAG Chat with PDF", page_icon="📄", layout="wide")

st.title('📄 RAG with PDF upload and chat history')

# --- Sidebar for API Key and Session ID ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Use different keys for widgets in sidebar to avoid conflict if same labels are used elsewhere
    api_key_sb = st.text_input('Enter your Gemini API key:', type='password', key='api_key_sidebar')
    session_id_sb = st.text_input('Session ID', value='default_session', key='session_id_sidebar')
    st.markdown("---")
    st.caption("Provide your API key and a session ID. Chats are stored per session.")

# Use the sidebar inputs for the main logic
api_key = api_key_sb
session_id = session_id_sb

if api_key:
    # Get model and its embedding
    model = GoogleGenerativeAI(google_api_key=api_key, model='models/gemini-2.0-flash', max_output_tokens=500)
    embedding = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model='models/text-embedding-004')
    
    # Initialize store for session histories if it doesn't exist
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    # Ensure history object exists for the current session_id before trying to access messages
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()

    # --- PDF Upload Section ---
    st.subheader("📤 1. Upload Your PDF Documents")
    pdf_files = st.file_uploader('Select PDF files to process:', type='pdf', accept_multiple_files=True, key='pdf_uploader_main')
    
    if pdf_files:
        # Show a spinner during document processing
        with st.spinner("⏳ Processing PDF(s)... This may take a moment."):
            documents = []
            for pdf_file in pdf_files: 
                temp_pdf = './temp.pdf'
                with open(temp_pdf, 'wb') as file:
                    file.write(pdf_file.getvalue())
                    
                loader = PyPDFLoader(temp_pdf)
                docs = loader.load()
                documents.extend(docs)

        # Rag chain settup
        context_prompt = (
            "Given a chat history and the latest user question"
            " which might reference context in the chat history,"
            " formulate a standalone question which can be understood without the chat history."
            " Do NOT answer the question,"
            " just reformulate it if needed and otherwise return it as is."
        )
        history_retriever = history_aware_retriever(documents, embedding, context_prompt, model)
        
        system_prompt = (
            "You are a helpful and conversational assistant for question-answering tasks. "
            "Given the chat history with the user and the following pieces of retrieved context from documents, "
            "answer the user's latest question. "
            "If the answer is not found in the retrieved context, say that you don't know. "
            "Keep your answer concise and within three sentences."
            "\n\n"
            "Retrieved Context:\n{context}"
        )
        qa_chain = question_answer_chain(system_prompt, model)
        
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)
        
        def get_session_history(session_param: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )
        
        # --- Chat Interface ---
        st.markdown("---") # Visual separator
        st.subheader(f"💬 2. Chat with: {', '.join([f.name for f in pdf_files])}")

        # Display existing chat messages from history
        chat_container = st.container() # Use a container for better layout if chat gets long
        with chat_container:
            if st.session_state.store[session_id].messages:
                for msg in st.session_state.store[session_id].messages:
                    role = "user" if msg.type == "human" else "assistant"
                    with st.chat_message(role):
                        st.markdown(msg.content)
        # User input
        user_new_query = st.chat_input(f"Ask a question about the content of your PDF(s)...")
        
        if user_new_query:
            with st.spinner("🤖 Assistant is thinking..."):
                response = conversational_rag_chain.invoke(
                    {'input': user_new_query},
                    config={'configurable': {'session_id': session_id}}
                )
            st.rerun()

    elif api_key and not pdf_files: # API key provided, but no PDFs selected yet
        st.info("⬆️ Please upload PDF document(s) to begin chatting.")

else:
    st.warning('Please enter your Gemini API key in the sidebar to use the application.')