import os
import streamlit as st
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.title("SimplifyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Use persistent Chroma DB directory
CHROMA_DIR = "./chroma_db"

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading data...✅")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Splitting text...✅")
    docs = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
    vectorstore.persist()  # Save to disk
    main_placeholder.text("Embeddings stored successfully...✅")

query = main_placeholder.text_input("Question: ")

if query:
    # Load Chroma DB from disk
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    result = chain({"question": query}, return_only_outputs=True)

    st.header("Answer")
    st.write(result["answer"])

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        for source in sources.split("\n"):
            st.write(source)
