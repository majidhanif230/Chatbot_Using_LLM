# 📚 Chat with Books - AI-Powered PDF Chatbot

Welcome to the **Chat with Books** project, an advanced AI-powered PDF chatbot designed to provide detailed and accurate responses based on the context extracted from PDF documents. This project leverages the power of **Generative AI** and **Retrieval-Augmented Generation (RAG)** to ensure that the responses are not only contextually relevant but also backed by precise information from the source documents.

## 📝 Overview

The objective of this project is to develop a chatbot that offers comprehensive answers to user queries by analyzing the content of uploaded PDFs. The bot efficiently handles large documents, breaks down the content into manageable chunks, and retrieves the most relevant passages to generate responses. Each response is accompanied by a citation that specifies the PDF and page number from which the information was derived. If the information is not found in the PDFs, the chatbot will indicate this and provide a contact number for further inquiries.

## 🚀 Features

- **Generative AI & RAG Integration**: Combines the capabilities of Generative AI and RAG to ensure responses are accurate and supported by the relevant information from the PDFs.
- **Document Parsing**: Efficiently processes large PDF documents, breaking down the text into chunks for better analysis.
- **Contextual Responses**: The chatbot is designed to retrieve and generate responses that are contextually relevant, providing accurate and detailed answers.
- **Source Attribution**: Every response includes a citation indicating the source PDF and the exact page number where the information was found.
- **User-Friendly Interface**: Utilizes Streamlit to create an interactive and easy-to-use interface.

## ⚙️ How It Works

1. **PDF Text Extraction**: The bot extracts text from uploaded PDF files and organizes it into chunks.
2. **Semantic Search**: Text chunks are encoded into vectors using Sentence Transformers, enabling sophisticated semantic search capabilities.
3. **Vector Store Creation**: A FAISS index is created to allow fast retrieval of relevant text passages based on user queries.
4. **Conversational Chain**: The chatbot generates answers by using the retrieved passages, ensuring that the responses are accurate and context-aware.
5. **User Input Handling**: Users can ask questions, and the bot processes the query to provide the most relevant response along with source attribution.

## 🛠️ Technologies Used

- **Streamlit**: For creating a web-based interactive user interface.
- **PyPDF2**: For extracting text from PDF files.
- **LangChain**: For managing text chunking, embedding, and conversational chains.
- **Google Generative AI**: For generating responses.
- **FAISS**: For efficient similarity search and passage retrieval.
- **Sentence Transformers**: For encoding text into vectors.

## 🚀 Getting Started

1. Clone the repository:
   git clone https://github.com/yourusername/chat-with-books.git
   cd chat-with-books
2. pip install -r requirements.txt
3. streamlit run app.py
## 🙏 Acknowledgements
This project was provided by Machine Learning 1 Pvt Ltd and completed by Majid Hanif. Special thanks to the open-source community and the developers of the libraries used in this project for their invaluable contributions.

