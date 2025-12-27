# 📄 PDF Chatbot using LangChain and Chainlit

An interactive **PDF-based conversational chatbot** that allows users to upload PDF documents and ask natural language questions to retrieve accurate, context-aware answers.  
The system is built using **Retrieval-Augmented Generation (RAG)** with a clean ChatGPT-like interface powered by **Chainlit**.

---

## Features
- Upload any PDF and chat with its content
- Real-time conversational interface (ChatGPT-like UI)
- Automatic text extraction and chunking
- Vector-based similarity search using FAISS
- Transformer-based language model for answer generation
- Supports multiple questions on the same document

---

## How It Works (RAG Pipeline)
1. User uploads a PDF document  
2. Text is extracted and split into manageable chunks  
3. Each chunk is converted into embeddings  
4. Embeddings are stored in a FAISS vector database  
5. Relevant chunks are retrieved using similarity search  
6. A transformer-based LLM generates answers using retrieved context  

---

## Technologies Used
- Python 3.11  
- LangChain  
- Chainlit  
- FAISS  
- HuggingFace Transformers  
- Sentence-Transformers  
- PyPDF  

---

## How to Run the Project

### 1️ Clone the repository
```bash
git clone (https://github.com/Balaji61/PDF_CHATBOT_CHAINLIT)
cd PDF_CHATBOT_CHAINLIT
```
### 2️ Create and activate virtual environment (Python 3.11)
```
py -3.11 -m venv venv
cmd /k venv\Scripts\activate.bat
```

### 3️ Install dependencies
```
pip install -r requirements.txt
```

### 4️ Run the application
```
chainlit run app.py
```
