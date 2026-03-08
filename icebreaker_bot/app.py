import gradio as gr
from pypdf import PdfReader
from dotenv import load_dotenv
import os

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq


# load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

index = None


def process_resume(file):
    global index

    if file is None:
        return "Please upload a resume."

    reader = PdfReader(file.name)

    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    if text.strip() == "":
        return "Could not extract text from PDF."

    documents = [Document(text=text)]

    # local embedding model
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

    return "✅ Resume processed successfully. Now ask questions."


def ask_question(question):
    global index

    if index is None:
        return "⚠️ Please process resume first."

    llm = Groq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    query_engine = index.as_query_engine(llm=llm)

    response = query_engine.query(question)

    return str(response)


with gr.Blocks() as demo:

    gr.Markdown("# 📄 Resume AI Assistant (Groq + LlamaIndex)")
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
