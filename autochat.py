import os
import streamlit as st
from langchain_upstage import ChatUpstage as Chat
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain import LLMChain, PromptTemplate
import time

RAG_API_PUBLIC_MODEL_NAME = st.secrets["RAG_API_PUBLIC_MODEL_NAME"]
RAG_API_BASE_URL = st.secrets["RAG_API_BASE_URL"]
rag_api_chat = Chat(model=RAG_API_PUBLIC_MODEL_NAME, base_url=RAG_API_BASE_URL)

solar_pro = Chat(model="solar-pro")

rag_prompt_template = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder("chat_history"),
        ("human", "{user_query}"),
    ]
)


def get_response(user_query):
    chain = rag_prompt_template | rag_api_chat | StrOutputParser()
    return chain.stream(
        {
            "chat_history": [],  # Pass an empty list for chat history
            "user_query": user_query,
        }
    )


def generate_question(content, llm=solar_pro):
    # Create a prompt template for generating a question
    question_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """ You are a helpful assistant that generates questions based on given content.",
         Given the following content:\n\n{content}
        
        Generate a single, concise question that can be answered based on the content above.
        The question should be one line long and not include any additional text or explanations.
        Be direct and to the point.""",
            )
        ]
    )

    # Create a chain using the RAG API and the question prompt template
    question_chain = question_prompt_template | llm | StrOutputParser()

    # Generate the question using the chain and stream the output
    return question_chain.stream({"content": content})


def rate_answer(content, answer, llm=solar_pro):
    # Create a prompt template for rating the answer
    rating_prompt_template = PromptTemplate(
        input_variables=["content", "answer"],
        template="""Given the following content:
        {content}
        
        And the following answer:
        {answer}
        
        Rate the answer on a scale of 1 to 5 based on how well it answers a question that could be asked about the content. 
        The rating should be based solely on the information provided in the content. 
        If the answer is not grounded in the content or is incorrect based on the content, give it a low rating.
        Respond with only a single number representing the rating, nothing else.
        We will to python int() to convert the rating to an integer, 
        so make sure to follow the instructions exactly and give a rating between 1 and 5.
        """,
    )

    # Create a LLMChain using the RAG API and the rating prompt template
    rating_chain = rating_prompt_template | llm | StrOutputParser()

    # Generate the rating using the LLMChain, trying up to 3 times
    for attempt in range(3):
        rating = rating_chain.invoke({"content": content, "answer": answer})
        try:
            return int(rating.strip())
        except ValueError:
            if attempt < 2:
                continue
            else:
                return -1

    return -1


if __name__ == "__main__":
    st.set_page_config(page_title="Sung's Paper Chat", page_icon="☀️", layout="wide")

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

    directory = "dataset_en"
    file_list = [file for file in os.listdir(directory) if file.endswith(".txt")]

    if st.button("Process All Files", type="primary"):
        progress_bar = st.progress(0)
        file_count = len(file_list)

        for index, file_name in enumerate(file_list):
            file_path = os.path.join(directory, file_name)
            with st.expander(f"Processing: {file_name}", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Content")
                    with open(file_path, "r") as f:
                        content = f.read()
                    st.text_area("", value=content, height=200, disabled=True)

                with col2:
                    st.subheader("Question")
                    question = st.write_stream(generate_question(content))

                    st.subheader("Answer")
                    answer = st.write_stream(get_response(question))

                    rating = rate_answer(content, answer)
                    st.metric("Rating", f"{rating}/5")

                    if "ratings" not in st.session_state:
                        st.session_state.ratings = []
                    if rating != -1:
                        st.session_state.ratings.append(rating)

            progress_bar.progress((index + 1) / file_count)
            time.sleep(0.5)  # Add a small delay for visual effect

        st.success("All files have been processed!")

        if "ratings" in st.session_state:
            valid_ratings = [r for r in st.session_state.ratings if r != -1]
            if valid_ratings:
                avg_rating = sum(valid_ratings) / len(valid_ratings)
                st.metric("Average Rating", f"{avg_rating:.2f}/5")
            else:
                st.warning("No valid ratings available.")
