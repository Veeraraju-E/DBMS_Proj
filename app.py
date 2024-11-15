import gradio as gr
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

# Function to generate responses
def generate_response(question):
    # Pure LLM Response
    inputs = llm_tokenizer(question, return_tensors="pt")
    outputs = llm_model.generate(**inputs)
    pure_llm_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # RAG Response
    question_embedding = vector_model.encode([question])
    _, top_k_indices = index.search(np.array(question_embedding), k=1)
    retrieved_context = documents[top_k_indices[0][0]]["text"]

    context_enhanced_input = f"Question: {question} Context: {retrieved_context}"
    inputs = llm_tokenizer(context_enhanced_input, return_tensors="pt")
    outputs = llm_model.generate(**inputs)
    rag_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pure_llm_response, retrieved_context, rag_response

# Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs=[
        gr.Textbox(label="Pure LLM Output"),
        gr.Textbox(label="Retrieved Context"),
        gr.Textbox(label="RAG Output (with context)")
    ],
    title="RAG Application Demo",
    description="This application demonstrates the difference between LLM outputs with and without context augmentation. Enter your question below!"
)

iface.launch()
