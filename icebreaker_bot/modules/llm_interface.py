import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq

load_dotenv()


def create_llm():

    llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

    return llm
