# 📄 Resume AI Assistant — Icebreaker Bot

> An intelligent, LLM-powered resume analysis tool that lets you upload any PDF résumé and ask natural-language questions about the candidate — built with **LlamaIndex**, **Groq**, **HuggingFace Embeddings**, and **Gradio**.

---

## 🚀 Features

- **PDF Resume Upload** — Drop any PDF résumé directly into the interface.
- **Automatic Text Extraction** — Extracts raw text from every page using `pypdf`.
- **Semantic Chunking & Indexing** — Splits text into meaningful chunks and builds a vector index with `SentenceSplitter` + HuggingFace embeddings (`all-MiniLM-L6-v2`).
- **Persistent Index Storage** — The vector index is saved locally (`./storage`) and reloaded on subsequent sessions — no reprocessing required.
- **LLM-Powered Q&A** — Queries the index with a custom prompt using Groq's `llama-3.1-8b-instant` model for fast, accurate answers.
- **Clean Gradio UI** — A simple, browser-based interface with upload, process, and ask-question controls.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | [Gradio](https://www.gradio.app/) |
| RAG Framework | [LlamaIndex](https://www.llamaindex.ai/) |
| LLM | [Groq — llama-3.1-8b-instant](https://groq.com/) |
| Embeddings | [HuggingFace — all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| PDF Parsing | [pypdf](https://pypdf.readthedocs.io/) |
| Environment Config | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## 📁 Project Structure

```
llamaindex_project/
└── icebreaker_bot/
    ├── app.py            # Main Gradio application
    ├── test.py           # Quick LLM connectivity test
    ├── requirements.txt  # Python dependencies
    └── .gitignore        # Git ignore rules
```

---

## ⚙️ Getting Started

### Prerequisites

- Python **3.9+**
- A [Groq API key](https://console.groq.com/) (free tier available)

### 1. Clone the Repository

```bash
git clone https://github.com/Bharath-kumar-7/llamaindex_project.git
cd llamaindex_project/icebreaker_bot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the `icebreaker_bot/` directory:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> ⚠️ Never commit your `.env` file. It is already listed in `.gitignore`.

### 5. Run the Application

```bash
python app.py
```

The Gradio interface will launch and print a local URL (e.g. `http://127.0.0.1:7860`) as well as a shareable public link.

---

## 🧭 How to Use

1. **Upload Resume** — Click the *Upload Resume PDF* button and select a PDF file.
2. **Process Resume** — Click **Process Resume**. Wait for the *"✅ Resume processed successfully!"* status message.
3. **Ask Questions** — Type any question in the *Your Question* box and click **Ask**.

### Example Questions

```
What are this candidate's top technical skills?
How many years of experience does the candidate have?
What is the candidate's highest level of education?
Has the candidate worked with Python?
Which companies has the candidate worked for?
```

---

## 🧪 Testing LLM Connectivity

To verify your Groq API key and model access independently:

```bash
python test.py
```

Expected output: a short greeting from the LLM (e.g., `Hello! How can I help you today?`).

---

## 📦 Dependencies

```
llama-index
llama-index-llms-groq
llama-index-embeddings-huggingface
sentence-transformers
python-dotenv
requests
```

Install all at once:

```bash
pip install -r requirements.txt
```

> **Note:** `pypdf` and `gradio` are pulled in transitively. If you encounter any missing package errors, run `pip install pypdf gradio`.

---

## 🔐 Environment Variables

| Variable | Description | Required |
|---|---|---|
| `GROQ_API_KEY` | API key for Groq LLM access | ✅ Yes |

---

## 🗺️ How It Works

```
                  ┌─────────────────────────────────┐
                  │         Gradio UI (Browser)      │
                  └───────────┬─────────────┬────────┘
                              │             │
                    Upload PDF │             │ Ask Question
                              ▼             ▼
                     ┌──────────────┐  ┌──────────────────┐
                     │  PDF Parser  │  │   Query Engine    │
                     │  (pypdf)     │  │   (LlamaIndex)    │
                     └──────┬───────┘  └────────┬─────────┘
                            │                   │
                            ▼                   ▼
                  ┌──────────────────┐  ┌───────────────────┐
                  │  Sentence        │  │  Groq LLM          │
                  │  Splitter +      │  │  (llama-3.1-8b)    │
                  │  HF Embeddings   │  └───────────────────┘
                  └──────┬───────────┘
                         │
                         ▼
                  ┌──────────────────┐
                  │  VectorStoreIndex│
                  │  (./storage)     │
                  └──────────────────┘
```

1. The PDF is parsed and split into text chunks.
2. Each chunk is embedded using HuggingFace's `all-MiniLM-L6-v2` model.
3. Embeddings are stored in a `VectorStoreIndex` and persisted to disk.
4. On a question, the top-3 most relevant chunks are retrieved and passed — along with the question — to Groq's Llama model.
5. The LLM returns a concise, context-grounded answer.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to your fork: `git push origin feature/your-feature-name`
5. Open a Pull Request.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

**Bharath Kumar**
- GitHub: [@Bharath-kumar-7](https://github.com/Bharath-kumar-7)

---

<p align="center">Built with ❤️ using LlamaIndex + Groq + Gradio</p>
