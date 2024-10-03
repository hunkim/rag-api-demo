import os
import json
from u2s import text2kvpairs, text2kg
from collections import defaultdict
import time
import streamlit as st
from langchain_upstage import ChatUpstage as Chat
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt

# Initialize ChatUpstage
default_llm = Chat(model="solar-pro")
solar_mini_llm = Chat(model="solar-1-mini-chat")

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from langchain_upstage import ChatUpstage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser




class SolarGenBaseClass(ABC):
    def __init__(self, name: str, context: str):
        self.name = name
        self.context = context
        self.generation: Optional[str] = None

    def generate(
        self,
        input: str,
        llm: ChatUpstage,
        **kwargs: Dict[str, Any]
    ) -> str:
        prompt_template = ChatPromptTemplate.from_messages([
            ("user", """{input}
             ---
             Context:
             {context}
             """),
        ])

        chain = prompt_template | llm | StrOutputParser()

        return chain.invoke({"input": input, "context": self.context})



def get_judge_score(context, question, answer, answer_likert_scale_prompt, max_attempts=3):
    for attempt in range(max_attempts):
        score_judge_pro = SolarGenBaseClass(name="score_judge_pro", context=context + "\n\nQuestion: " + question + "\n\nAnswer: " + answer)
        score_judge_mini = SolarGenBaseClass(name="score_judge_mini", context=context + "\n\nQuestion: " + question + "\n\nAnswer: " + answer)
        
        score_text_pro = score_judge_pro.generate(answer_likert_scale_prompt, llm=default_llm)
        score_text_mini = score_judge_mini.generate(answer_likert_scale_prompt, llm=solar_mini_llm)
        
        try:
            score_pro = int(score_text_pro.strip())
            score_mini = int(score_text_mini.strip())
            
            if 1 <= score_pro <= 5 and 1 <= score_mini <= 5:
                if score_pro == score_mini:
                    return score_pro
                elif attempt == max_attempts - 1:
                    return (score_pro + score_mini) / 2
            else:
                raise ValueError("Score out of range")
        except ValueError:
            if attempt < max_attempts - 1:
                print(f"Invalid scores: Pro: {score_text_pro}, Mini: {score_text_mini}. Retrying... (Attempt {attempt + 2}/{max_attempts})")
            else:
                print(f"Failed to get valid scores after {max_attempts} attempts. Using average of last attempt.")
                return (score_pro + score_mini) / 2 if 'score_pro' in locals() and 'score_mini' in locals() else 0
    
    return 0

def main():
    st.set_page_config(page_title="Solar RAG Benchmark", page_icon="☀️", layout="wide")

    st.markdown(
        """
        <h1 style='text-align: center; color: #FFA500;'>
            ☀️ Solar RAG Benchmark
        </h1>
        <h3 style='text-align: center; color: #4A4A4A;'>
            Evaluate RAG performance with various context types
        </h3>
        """,
        unsafe_allow_html=True,
    )

    question_generation_prompt = """
    Given the context, generate a single-line, challenging question that requires precise understanding of the information provided. The question should:
    1. Focus on specific facts, numbers, dates, or locations explicitly mentioned in the context.
    2. Test the ability to locate and extract exact information from the text.
    3. Require combining multiple pieces of information directly stated in the context.
    4. Be clear and concise, while still being difficult to answer without a thorough reading of the context.
    5. Avoid general or easily answerable questions.
    6. Ensure that the answer is explicitly stated in the context, without requiring any inference or external knowledge.
    
    Aim to create questions that would challenge an AI's ability to accurately locate and retrieve detailed information directly from the given context. Do not ask questions that require inferencing or interpretation beyond what is explicitly stated.
    """

    answer_generation_prompt = """
    You are an expert in providing accurate and concise answers based on the given context. Your task is to generate a clear, factual answer to the provided question using only the information available in the context. Follow these guidelines:

    1. Use only the information present in the context to answer the question.
    2. Provide a direct and concise answer without adding any external knowledge.
    3. If the question cannot be fully answered using the context, state what information is available and what is missing.
    4. Ensure the answer is factually correct and directly related to the question.
    5. If the context doesn't contain relevant information to answer the question, respond with "The given context does not provide sufficient information to answer this question."

    Remember, the goal is to create a reliable truth dataset for evaluating LLMs and RAG systems, so accuracy and relevance are crucial.
    """

    answer_likert_scale_prompt = """
    As an expert evaluator, your task is to strictly rate the given answer based on its accuracy, relevance, correctness (precision), and completeness (recall) in relation to the question and context. Use the following rigorous 5-point Likert scale:

    1 - Poor: The answer is incorrect, incomplete, or irrelevant.
    2 - Fair: The answer is partially correct but lacks significant information, contains inaccuracies, or is not sufficiently complete.
    3 - Good: The answer is mostly correct and relevant, with minor omissions or slight inaccuracies. It addresses most key points but may not be fully comprehensive.
    4 - Very Good: The answer is correct, relevant, and comprehensive, with only trivial imperfections. It demonstrates high precision and recall, missing only minor details.
    5 - Excellent: The answer is perfectly accurate, highly relevant, and complete. It exhibits flawless precision and recall, covering all necessary information from the context.

    Provide only the numeric score (1-5) as your response. Be strict in your evaluation, reserving higher scores for truly exceptional answers.
    We will use python int() to convert your response to an integer, so ensure you return only the number.
    """

    if st.button("Run Benchmark", type="primary"):
        scores = defaultdict(list)
        dataset_folder = "dataset_en"
        file_paths = [os.path.join(dataset_folder, file) for file in os.listdir(dataset_folder) if file.endswith(".txt")]
        total_files = len(file_paths)

        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        start_time = time.time()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            overall_progress = st.empty()
        with col2:
            time_elapsed = st.empty()
        with col3:
            st.markdown("### Average Scores")
            avg_score_metrics = {
                "context_only": st.empty(),
                "context_with_kv": st.empty(),
                "context_with_kg": st.empty(),
                "context_with_kv_and_kg": st.empty()
            }

        for file_index, file_path in enumerate(file_paths, 1):
            file_start_time = time.time()
            
            with open(file_path, 'r') as file:
                context = file.read()

            with st.expander(f"Processing file {file_index}/{total_files}: {os.path.basename(file_path)}", expanded=False):
                # Process different context types
                st.info("Generating Key-Value Pairs...")
                kv = text2kvpairs(context, llm=default_llm)
                st.json(kv)
                
                st.info("Generating Knowledge Graph...")
                kg = text2kg(context, kv, llm=default_llm)
                st.json(kg)

                context_types = {
                    "context_only": context,
                    "context_with_kv": context + "\n\nKey-Value Pairs:\n" + json.dumps(kv, ensure_ascii=False),
                    "context_with_kg": context + "\n\nKnowledge Graph:\n" + json.dumps(kg, ensure_ascii=False),
                    "context_with_kv_and_kg": context + "\n\nKey-Value Pairs:\n" + json.dumps(kv, ensure_ascii=False) +
                                              "\n\nKnowledge Graph:\n" + json.dumps(kg, ensure_ascii=False)
                }

                # Generate question
                st.info("Generating Question...")
                question_gen = SolarGenBaseClass(name="question_generator", context=context)
                question = question_gen.generate(question_generation_prompt, llm=default_llm)
                st.write(f"**Question:** {question}")

                results = {}

                for context_type, enriched_context in context_types.items():
                    answer_gen = SolarGenBaseClass(name=f"{context_type}_generator", context=enriched_context + "\n\nQuestion: " + question)
                    try:
                        st.info(f"Generating Answer for {context_type}...")
                        answer = answer_gen.generate(answer_generation_prompt, llm=default_llm)
                        score = get_judge_score(context, question, answer, answer_likert_scale_prompt, max_attempts=3)
                        scores[context_type].append(score)
                        results[context_type] = {"answer": answer, "score": score}
                        st.success(f"{context_type.replace('_', ' ').title()}: Score {score}/5")
                    except Exception as e:
                        st.error(f"Error processing {context_type}: {e}")

            # Update progress and metrics
            progress_percentage = (file_index) / total_files
            progress_bar.progress(progress_percentage)
            overall_progress.metric("Overall Progress", f"{progress_percentage:.0%}")
            
            current_time = time.time()
            elapsed = current_time - start_time
            time_elapsed.metric("Time Elapsed", f"{elapsed:.2f}s")

            # Update average scores for each context type
            for context_type, score_list in scores.items():
                if score_list:
                    avg_score = sum(score_list) / len(score_list)
                    avg_score_metrics[context_type].metric(
                        f"{context_type.replace('_', ' ').title()}",
                        f"{avg_score:.2f}/5"
                    )
                else:
                    avg_score_metrics[context_type].metric(
                        f"{context_type.replace('_', ' ').title()}",
                        "N/A"
                    )

            # Display current results
            with results_container:
                st.subheader(f"Results for {os.path.basename(file_path)}")
                st.write(f"**Question:** {question}")
                for context_type, result in results.items():
                    with st.expander(f"{context_type.replace('_', ' ').title()} (Score: {result['score']}/5)"):
                        st.write(f"**Answer:** {result['answer']}")
                st.markdown("---")

        # Calculate and display final average scores and total time
        end_time = time.time()
        total_time = end_time - start_time

        st.success("Benchmark completed!")
        st.write(f"Total time elapsed: {total_time:.2f} seconds")

        st.subheader("Final Average Scores")
        chart_data = []
        for context_type, score_list in scores.items():
            avg_score = sum(score_list) / len(score_list)
            chart_data.append({"Context Type": context_type.replace('_', ' ').title(), "Average Score": avg_score})
        
        st.bar_chart(chart_data, x="Context Type", y="Average Score")

        # Display detailed statistics
        st.subheader("Detailed Statistics")
        for context_type, score_list in scores.items():
            with st.expander(context_type.replace('_', ' ').title()):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Average Score", f"{sum(score_list) / len(score_list):.2f}")
                col2.metric("Minimum Score", min(score_list))
                col3.metric("Maximum Score", max(score_list))
                col4.metric("Total Samples", len(score_list))
                st.pyplot(plt.figure(figsize=(10, 5)))
                plt.hist(score_list, bins=5, edgecolor='black')
                plt.title(f"Score Distribution for {context_type.replace('_', ' ').title()}")
                plt.xlabel("Score")
                plt.ylabel("Frequency")
                st.pyplot(plt)

if __name__ == "__main__":
    main()