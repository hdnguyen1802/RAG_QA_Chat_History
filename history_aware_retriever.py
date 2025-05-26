from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
def history_aware_retriever(documents, embedding, context_prompt:str, model):
    # Create a retriever
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 6666, chunk_overlap = 666)
    split_document = splitter.split_documents(documents)
    vector_store = Chroma.from_documents(split_document,embedding)
    retriever = vector_store.as_retriever()
    
    # Create a contexualize_prompt
    
    context_prompt = ChatPromptTemplate.from_messages(
        [
            ('system',context_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ]
    )
    retriever_with_memory = create_history_aware_retriever(model, retriever, context_prompt)
    
    return retriever_with_memory