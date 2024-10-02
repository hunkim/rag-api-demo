from openai import OpenAI
import streamlit as st
import os


RAG_API_BASE_URL = st.secrets["RAG_API_BASE_URL"]
RAG_API_MODEL_NAME = st.secrets["RAG_API_MODEL_NAME"]
API_KEY = st.secrets["UPSTAGE_API_KEY"]
client = OpenAI(base_url=RAG_API_BASE_URL, api_key=API_KEY)

print("Building/Rebuilding the index...")
file_response = None
directory = "dataset_en"
# Upload all files in the pdfs folder
for file in os.listdir(directory):
    file_path = os.path.join(directory, file)
    print(file_path)
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            file_response = client.files.create(
            file=f, purpose="assistants", extra_body={"model_name": RAG_API_MODEL_NAME}
        )
        print(file_response)

print("Index build/rebuild submission completed.")

if file_response:
    public_model_name = file_response.public_model_name
    print("Public Mode Name:", public_model_name)
else:
    print("File upload failed.")    

print("Index build/rebuild submission completed.")