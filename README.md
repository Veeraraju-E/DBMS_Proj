# RAG Application Demo

This Gradio application demonstrates the difference between:

- **Pure LLM Output**: Generated directly from the model without additional context.
- **RAG Output**: Generated with relevant context retrieved from a vector database.

## How to Use

1. Enter a question in the input box.
2. The application will display:
   - Pure LLM output.
   - Retrieved context.
   - RAG output with context enhancement.

## Powered by

- Hugging Face Transformers
- SentenceTransformers
- FAISS
