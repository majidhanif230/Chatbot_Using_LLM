# Chat with Books — AI-Powered PDF Chatbot

A Streamlit chatbot that answers questions from your own PDF documents using Generative AI and Retrieval-Augmented Generation (RAG). Upload PDFs, ask a question, and get a detailed answer drawn from the documents — with a citation naming the source PDF and page number. If the answer is not in the documents, the bot says so instead of inventing one.

## Key Features

- **Chat with your PDFs** — upload one or more PDFs in the sidebar and ask questions about their contents.
- **RAG pipeline** — answers are generated from retrieved document passages, keeping them grounded in your files.
- **Source attribution** — every answer cites its source, e.g. *"This information is taken from \<pdf name\> on page \<page number\>."*
- **Honest fallback** — when the information is not in the provided context, the bot replies that it doesn't have the information and gives a contact number for further enquiries.
- **Streamlit UI** — simple chat interface with a file uploader and a "Submit & Process" step that builds the search index.
- **Bonus notebook** — `nb/Gemma3_(270M).ipynb` fine-tunes Google's Gemma-3-270M model with LoRA (see below).

## How It Works

1. **PDF text extraction** — `PyPDF2` reads each uploaded PDF, tracking the source file name and page number of every passage.
2. **Text chunking** — `RecursiveCharacterTextSplitter` splits the text into 1000-character chunks with 200 characters of overlap.
3. **Embeddings** — chunks are embedded with Google's `models/embedding-001` model.
4. **Vector store** — embeddings are stored in a local FAISS index (`faiss_index/`), rebuilt each time PDFs are processed.
5. **Retrieval** — a question is embedded and the most similar chunks are pulled from FAISS.
6. **Answer generation** — a LangChain QA chain (`load_qa_chain`, `stuff` type) with `ChatGoogleGenerativeAI` (`gemini-pro`, temperature 0.8) and a custom prompt answers from the retrieved context and appends the source citation.

### Gemma-3 fine-tuning notebook (`nb/Gemma3_(270M).ipynb`)

A separate Colab notebook that fine-tunes `unsloth/gemma-3-270m-it` with LoRA (rank 128, alpha 128) using Unsloth, on the first 10,000 examples of the `Thytu/ChessInstruct` dataset. Training uses the TRL `SFTTrainer` (100 steps, batch size 8, learning rate 5e-5, max sequence length 2048), trains on assistant responses only, and saves the LoRA adapters locally (`gemma-3/`), with optional merge/export to 16-bit, 4-bit, or GGUF. In the recorded run, training took about 8.3 minutes on a free Tesla T4.

## Tech Stack

- **Python**, **Streamlit** — web interface
- **LangChain** — text splitting, QA chain, prompt templates
- **Google Generative AI** — Gemini Pro for answers, `embedding-001` for embeddings
- **FAISS** (`faiss-cpu`) — vector similarity search
- **PyPDF2** — PDF text extraction
- **Unsloth, TRL, PEFT** — LoRA fine-tuning in the Gemma-3 notebook

## How to Run

```bash
git clone https://github.com/majidhanif230/Chatbot_Using_LLM.git
cd Chatbot_Using_LLM
pip install -r requirements.txt.txt
```

1. Set your Google API key (the app uses `GOOGLE_API_KEY` for Gemini and embeddings).
2. Start the app:

```bash
streamlit run Chatbot_Using_LLM.py
```

3. In the sidebar, upload your PDFs and click **Submit & Process**. Then type a question in the main panel to chat with your documents.

To run the Gemma-3 notebook, open `nb/Gemma3_(270M).ipynb` in Google Colab (free Tesla T4) and run all cells.

## Project Structure

```
Chatbot_Using_LLM/
├── Chatbot_Using_LLM.py        # Streamlit app: PDF upload, FAISS indexing,
│                               # Gemini-powered QA chain with source citations
├── requirements.txt.txt        # streamlit, PyPDF2, langchain, langchain-google-genai,
│                               # google-generativeai, faiss-cpu
├── nb/
│   └── Gemma3_(270M).ipynb    # LoRA fine-tuning of Gemma-3-270M on ChessInstruct (Unsloth)
├── .devcontainer/              # Dev container configuration
└── README.md
```

## Acknowledgements

Project provided by Machine Learning 1 Pvt Ltd; completed by Majid Hanif.

## Author

**Majid Hanif** — AI Engineer · BS Artificial Intelligence, The Islamia University of Bahawalpur (CGPA 3.81/4.0)

- Portfolio: https://snap-portfolio-me.lovable.app/portfolio
- LinkedIn: https://www.linkedin.com/in/majid-hanif-23720b2a2/
- GitHub: https://github.com/majidhanif230
