from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

def question_answer_chain(system_prompt, model):
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),
            MessagesPlaceholder('chat_history'),
            ('human','{input}')
        ]
    )
    qa_chain = create_stuff_documents_chain(model, qa_prompt)
    
    return qa_chain