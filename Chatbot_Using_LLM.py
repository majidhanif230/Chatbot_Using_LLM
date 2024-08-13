
# --- Libraries Import ---
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# --- API Key Setup ---
GOOGLE_API_KEY = "Add Your Key"
genai.configure(api_key=GOOGLE_API_KEY)

# --- PDF Text Extraction Function ---
def get_pdf_text(pdf_docs):
    text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_number, page in enumerate(pdf_reader.pages):
            extracted_text = page.extract_text()
            if extracted_text:
                text.append({
                    'content': extracted_text,
                    'pdf_name': os.path.basename(pdf.name),
                    'page_number': page_number 
                })
    return text

# --- Text Chunking Function ---
def get_text_chunks(pdf_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for pdf_text in pdf_texts:
        split_chunks = text_splitter.split_text(pdf_text['content'])
        for chunk in split_chunks:
            chunks.append({
                'chunk': chunk,
                'pdf_name': pdf_text['pdf_name'],
                'page_number': pdf_text['page_number']
            })
    return chunks

# --- Vector Store Creation Function ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    documents = []

    for chunk in text_chunks:
        documents.append(Document(
            page_content=chunk['chunk'],
            metadata={
                'pdf_name': chunk['pdf_name'],
                'page_number': chunk['page_number']
            }
        ))

    vector_store = FAISS.from_documents(documents, embedding=embeddings)
    vector_store.save_local("faiss_index")

# --- Conversational Chain Setup Function ---
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in 
    provided context just say, "I don't have this information. For more information, contact +123456789." Don't provide the wrong answer.
    Also, with every response you provide, make sure to mention the source of the information in next line like "This information is taken from {pdf_name} on page {page_number}."\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "pdf_name", "page_number"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

# --- User Input Handling Function ---
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    context = "\n".join([doc.page_content for doc in docs])
    pdf_name = docs[0].metadata['pdf_name'] if docs and 'pdf_name' in docs[0].metadata else "Unknown"
    page_number = docs[0].metadata['page_number'] if docs and 'page_number' in docs[0].metadata else 0

    response = chain(
        {"input_documents": docs, "context": context, "question": user_question, "pdf_name": pdf_name, "page_number": page_number + 1},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

# --- Main Streamlit App Function ---
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Books📚")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Overview:")
        pdf_docs = st.file_uploader("The objective of this project is to develop a chatbot that provides detailed and accurate answers based on the context extracted from PDFs. If the information is not available in the context, the chatbot will indicate this and provide a contact number for further inquiries. Each response will also include a citation specifying the PDF and page number from which the information was derived.", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

# --- Run the Streamlit App ---
if __name__ == "__main__":
    main()