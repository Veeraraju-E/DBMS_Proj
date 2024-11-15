import gradio as gr
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize models
llm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
llm_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
vector_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Function to search the web and get relevant context
def search_web(query):
    """
    Fetch top web results for the query and return relevant context.
    """
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    # Parse search results using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")
    summaries = soup.find_all("div", class_="BNeawe s3v9rd AP7Wnd")
    
    web_context = []
    for res, summ in zip(results, summaries):
        if res and summ:
            web_context.append(f"{res.text}: {summ.text}")
    return " ".join(web_context[:3])  # Combine top 3 results as context

# Function to generate responses with and without web context
def generate_response(question):
    # Pure LLM Response
    inputs = llm_tokenizer(question, return_tensors="pt")
    outputs = llm_model.generate(**inputs)
    pure_llm_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Web Augmented Response (RAG)
    web_context = search_web(question)
    context_enhanced_input = f"Question: {question} Context: {web_context}"
    inputs = llm_tokenizer(context_enhanced_input, return_tensors="pt")
    outputs = llm_model.generate(**inputs)
    rag_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pure_llm_response, web_context, rag_response

# Gradio Interface
iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs=[
        gr.Textbox(label="Pure LLM Output"),
        gr.Textbox(label="Web Retrieved Context"),
        gr.Textbox(label="RAG Output (with context)")
    ],
    title="Web-Augmented RAG Application",
    description="Enter your question below to compare the LLM's response with and without web-based context augmentation."
)

iface.launch()
