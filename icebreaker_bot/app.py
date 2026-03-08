import gradio as gr
from pypdf import PdfReader
from dotenv import load_dotenv
import os

from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq


# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

index = None


# -----------------------------
# PROCESS RESUME
# -----------------------------
def process_resume(file):
    global index

    if file is None:
        return "⚠️ Please upload a resume."

    if not file.name.endswith(".pdf"):
        return "⚠️ Please upload a PDF file."

    try:
        reader = PdfReader(file.name)

        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        if text.strip() == "":
            return "⚠️ Could not extract text from the PDF."

        documents = [Document(text=text, metadata={"source": file.name})]

        # Embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder="./models"
        )

        # Split text into chunks
        splitter = SentenceSplitter(chunk_size=400)
        nodes = splitter.get_nodes_from_documents(documents)

        # Create vector index
        index = VectorStoreIndex(nodes, embed_model=embed_model)

        # Save index
        index.storage_context.persist("./storage")

        return "✅ Resume processed successfully! Ask your questions."

    except Exception as e:
        return f"❌ Error processing resume: {str(e)}"


# -----------------------------
# QUESTION ANSWERING
# -----------------------------
def ask_question(question):
    global index

    if question.strip() == "":
        return "⚠️ Please enter a question."

    # Load saved index if not loaded
    if index is None:

        if os.path.exists("./storage"):
            storage_context = StorageContext.from_defaults(persist_dir="./storage")
            index = load_index_from_storage(storage_context)

        else:
            return "⚠️ Please process a resume first."

    try:

        llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)

        qa_prompt = PromptTemplate(
            """
You are an AI assistant analyzing a candidate's resume.

Use ONLY the information from the resume context.

Resume Context:
{context_str}

Question:
{query_str}

Answer clearly and concisely.
"""
        )

        query_engine = index.as_query_engine(
            llm=llm, similarity_top_k=3, text_qa_template=qa_prompt
        )

        response = query_engine.query(question)

        return str(response)

    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


# -----------------------------
# GRADIO UI
# -----------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 📄 Resume AI Assistant")
    gr.Markdown("Upload a resume and ask questions about the candidate.")

    with gr.Row():

        resume_file = gr.File(label="Upload Resume PDF")
        process_btn = gr.Button("Process Resume")

    status = gr.Textbox(label="Status")

    process_btn.click(process_resume, inputs=resume_file, outputs=status)

    gr.Markdown("## Ask Questions")

    question = gr.Textbox(
        label="Your Question",
        placeholder="Example: What skills does this candidate have?",
    )

    ask_btn = gr.Button("Ask")

    answer = gr.Textbox(label="Answer")

    ask_btn.click(ask_question, inputs=question, outputs=answer)


demo.launch()
