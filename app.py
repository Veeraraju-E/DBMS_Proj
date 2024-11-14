from datasets import load_dataset
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import chromadb
from sentence_transformers import SentenceTransformer

class VectorStore:

    def __init__(self, collection_name) -> None:
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection_name)

    def populate_vectors(self, dataset) -> None:
        """
        Populate the vector store with embeddings from the dataset
        """
        for i, item in enumerate(dataset):
            combined_text = f"{item['instruction']} {item['context']}"
            embeddings = self.embedding_model.encode(combined_text).tolist()
            self.collection.add(embeddings=[embeddings], documents=[combined_text], ids=[f"id_{i}"])

    def search_context(self, query, n_result=1):
        """
        Search the ChromaDB collection to retrieve relevant context based on input query
        """
        query_embeddings = self.embedding_model.encode(query).tolist()
        result = self.collection.query(query_embeddings=query_embeddings, n_results=n_result)
        return result.get('documents', ["No context found."])[0]


def answer_question_with_context(question):
    # Get context from vector store
    context = vector_store.search_context(question)
    # Generate answer with context
    return gpt_2.generate_answer(question, context=context)

def answer_question_without_context(question):
    # Generate answer without context
    return gpt_2.generate_answer(question)


# Load and filter a smaller dataset
train_dataset = load_dataset("databricks/databricks-dolly-15k", split='train')
closed_qa_dataset = train_dataset.filter(lambda example: example['category'] == 'closed_qa')
closed_qa_dataset = closed_qa_dataset.select(range(100))  # Limit to first 100 samples

# Initialize VectorStore and populate with limited dataset
vector_store = VectorStore(collection_name="qa_collection")
vector_store.populate_vectors(closed_qa_dataset)

class GPT_2:

    def __init__(self):
        model_name = "gpt2"  # choose any another accessible model
        self.pipeline, self.tokenizer = self.initialize_model(model_name)

    def initialize_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        return pipeline, tokenizer


    def initialize_model(self, model_name):
        # Tokenizer initialization
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        return pipeline, tokenizer

    def generate_answer(self, question, context=None):
        # Preparing the input prompt with context if available
        prompt = question if context is None else f"Context: {context}\n\nQuestion: {question}"
        # Generate response with shortened length and reduced top_k
        sequences = self.pipeline(
            prompt,
            max_new_tokens=100,
            do_sample=True,
            top_k=5,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            truncation=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return sequences[0]['generated_text']

# Initialize model
gpt_2 = GPT_2()

# Gradio Interface
iface = gr.Interface(
    fn=lambda question: (answer_question_with_context(question), answer_question_without_context(question)),
    inputs=gr.Textbox(label="Enter your question"),
    outputs=[
        gr.Textbox(label="Answer with Context"),
        gr.Textbox(label="Answer without Context")
    ],
    title="Question Answering System",
    description="Ask a question and compare answers generated with and without additional context from our knowledge base."
)

# Launch the interface
iface.launch()
