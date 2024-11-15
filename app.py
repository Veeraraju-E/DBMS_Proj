import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load models
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
vector_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Initialize FAISS index and documents
index = faiss.IndexFlatL2(384)  # 384 is the dimension of embeddings
documents = [
    {"id": 1, "text": "The capital of France is Paris."},
    {"id": 2, "text": "Python is a popular programming language."},
    {"id": 3, "text": "Hugging Face provides open-source NLP tools."},
]
doc_embeddings = vector_model.encode([doc["text"] for doc in documents])
index.add(np.array(doc_embeddings))

# Streamlit UI
st.title("RAG Application Demo")
st.write("Compare LLM outputs with and without context augmentation!")

query = st.text_input("Enter your question:")

if st.button("Generate"):
    if query:
        # Pure LLM Response
        inputs = llm_tokenizer(query, return_tensors="pt")
        outputs = llm_model.generate(**inputs)
        pure_llm_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # RAG Response
        question_embedding = vector_model.encode([query])
        _, top_k_indices = index.search(np.array(question_embedding), k=1)
        retrieved_context = documents[top_k_indices[0][0]]["text"]

        context_enhanced_input = f"Question: {query} Context: {retrieved_context}"
        inputs = llm_tokenizer(context_enhanced_input, return_tensors="pt")
        outputs = llm_model.generate(**inputs)
        rag_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display results
        st.subheader("Pure LLM Output:")
        st.write(pure_llm_response)

        st.subheader("Retrieved Context:")
        st.write(retrieved_context)

        st.subheader("RAG Output (with context):")
        st.write(rag_response)
    else:
        st.warning("Please enter a question!")
