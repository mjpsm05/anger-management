import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import requests
import re

# --- Configuration ---
PINECONE_API_KEY = st.secrets["pinecone"]["api-key"]
INDEX_NAME = "anger-management-pdf"
HUGGINGFACE_API_TOKEN = st.secrets["huggingface"]["api-key"]
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

# --- Initialization ---
pinecone = Pinecone(api_key=PINECONE_API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = pinecone.Index(INDEX_NAME)

# --- Helper Function ---
def search_and_generate_answer(query):
    try:
        # Generate embedding
        query_embedding = embedder.encode(query).tolist()

        # Query Pinecone
        result = index.query(vector=query_embedding, top_k=3, include_metadata=True)

        # Extract full context (no truncation)
        context_chunks = [match["metadata"].get("text", "") for match in result.matches if "text" in match["metadata"]]
        context = " ".join(context_chunks)

        if not context:
            return "❌ Sorry, I couldn't find any relevant information."

        # Construct prompt
        prompt = f"""
        You are a helpful assistant. Based on the context, answer the user's question.

        Context:
        {context}

        Question: {query}

        Answer:
        """

        # Hugging Face API call
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1200,
                "temperature": 0.5
            }
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()

        if isinstance(response_data, list) and "generated_text" in response_data[0]:
            full_text = response_data[0]["generated_text"]

            # Extract answer using regex
            match = re.search(r"Answer:\s*(.*)", full_text, re.DOTALL)
            if match:
                return match.group(1).strip()
            else:
                return full_text.strip()
        else:
            return "⚠️ Unexpected response format from Hugging Face."

    except requests.exceptions.RequestException as e:
        return f"❌ API error: {e}"
    except Exception as e:
        return f"⚠️ Unexpected error: {e}"

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Anger Management Chatbot 😡")
st.subheader("Ask a question on how to manage your anger so you can be at peace 🙏🏾🙌🏾")

query = st.text_input("Enter your question")

if query:
    with st.spinner("Generating answer..."):
        answer = search_and_generate_answer(query)
        print(answer)
        st.markdown(answer.replace('\n', '  \n'))



