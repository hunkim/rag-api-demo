# from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import streamlit as st
from langchain_upstage import ChatUpstage as Chat

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from typing import List, Tuple

RAG_API_PUBLIC_MODEL_NAME = st.secrets["RAG_API_PUBLIC_MODEL_NAME"]
RAG_API_BASE_URL = st.secrets["RAG_API_BASE_URL"]
rag_api_chat = Chat(model=RAG_API_PUBLIC_MODEL_NAME, base_url=RAG_API_BASE_URL)

st.set_page_config(
    page_title="Sung's Paper Chat | Powered by Solar RAG API",
    page_icon="☀️",
    layout="wide",
)

st.markdown(
    """
    <h1 style='text-align: center; color: #FFA500;'>
        ☀️ Sung's Paper Chat
    </h1>
    <h3 style='text-align: center; color: #4A4A4A;'>
        Powered by Solar RAG API
    </h3>
    """,
    unsafe_allow_html=True,
)

rag_prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        ("human", "{user_query}"),
    ]
)


def get_three_questions() -> List[Tuple[str, str]]:
    """
    Returns a list of three tuples, each containing a button label and a question.
    """
    return [
        ("What is RAG?", "What is RAG in the context of AI and language models?"),
        ("How does Solar API work?", "Can you explain how the Solar RAG API works?"),
        (
            "Benefits of Solar API?",
            "What are the main benefits of using the Solar RAG API?",
        ),
    ]


def get_response(user_query, chat_history):
    chain = rag_prompt_template | rag_api_chat | StrOutputParser()

    response = ""
    end_token = ""
    return chain.stream(
        {
            "chat_history": chat_history,
            "user_query": user_query,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history outside the columns
for message in st.session_state.messages:
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(message.content)

st.markdown("### Quick Questions")
questions = get_three_questions()

col1, col2, col3 = st.columns(3)

for i, (button_text, question) in enumerate(questions):
    with [col1, col2, col3][i]:
        if st.button(button_text, key=f"btn_{i}"):
            st.session_state.messages.append(HumanMessage(content=question))
            st.session_state.current_question = question

# Handle the current question if there is one
if hasattr(st.session_state, "current_question"):
    question = st.session_state.current_question
    with st.chat_message("assistant"):
        response = st.write_stream(
            get_response(question, st.session_state.messages[:-1])
        )
    st.session_state.messages.append(AIMessage(content=response))
    del st.session_state.current_question

if prompt := st.chat_input("Ask a question or type your own"):
    # Uncomment and implement the enhance_prompt feature if needed
    # enhance_prompt = st.toggle("Enhance prompt", True)
    # if enhance_prompt:
    #     with st.status("Prompt engineering..."):
    #         new_prompt = prompt_engineering(prompt, st.session_state.messages)
    #         st.write(new_prompt)
    #     if "enhanced_prompt" in new_prompt:
    #         prompt = new_prompt["enhanced_prompt"]

    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = st.write_stream(get_response(prompt, st.session_state.messages))
    st.session_state.messages.append(AIMessage(content=response))

# ... rest of the existing code ...
