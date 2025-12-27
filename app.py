import chainlit as cl
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline

@cl.on_chat_start
async def start():
    files = await cl.AskFileMessage(
        content="📄 Please upload a PDF to begin",
        accept=["application/pdf"],
        max_size_mb=20,
    ).send()

    if not files:
        await cl.Message(content="No file uploaded. Please refresh and try again.").send()
        return

    pdf_file = files[0]
    file_path = pdf_file.path

    await cl.Message(content="⏳ Processing your PDF...").send()

    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # LOCAL LLM (stable)
    text_gen_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=text_gen_pipeline)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    cl.user_session.set("qa_chain", qa_chain)

    await cl.Message(content="✅ PDF processed! Ask your questions.").send()

@cl.on_message
async def main(message: cl.Message):
    qa_chain = cl.user_session.get("qa_chain")

    if not qa_chain:
        await cl.Message(content="⚠️ Please upload a PDF first.").send()
        return

    await cl.Message(content=" Thinking...").send()

    answer = await cl.make_async(qa_chain.run)(message.content)

    await cl.Message(content=answer).send()
