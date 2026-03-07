from llama_index.llms.groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

llm = Groq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

response = llm.complete("Say hello")

print(response)
