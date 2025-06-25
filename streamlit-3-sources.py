#Chatbot using LangChain and Streamlit
# This code demonstrates how to retrieve the information sources in 3 chunks
# and using 3 different sources.
# Run : streamlit run streamlit-3-sources.py

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore

import os
from langchain import hub

from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import bs4  # BeautifulSoup for parsing HTML

load_dotenv()  # take environment variables

# from .env file
# Load environment variables from .env file

token = os.getenv("SECRET_OPENAI")  # Replace with your actual token
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"

loader = WebBaseLoader(
    web_paths=("https://w.wiki/EXa6","https://www.britannica.com/place/Kaunas"),
    )
docs = loader.load()

# Load PDF file and add to docs
pdf_loader = PyPDFLoader("Kaunas-info.pdf")
pdf_docs = pdf_loader.load()
for doc in pdf_docs:
    doc.metadata['source'] = 'Kaunas-info.pdf'
docs.extend(pdf_docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
splits = text_splitter.split_documents(docs)

embeddings=OpenAIEmbeddings(
    model="text-embedding-3-small",
    base_url="https://models.inference.ai.azure.com",
    api_key=token, # type: ignore
)

vectorstore = InMemoryVectorStore(embeddings)

# Add splits to vectorstore one at a time to avoid token limit errors
for split in splits:
    vectorstore.add_documents([split])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

st.title("Streamlit LangChain Chatbot: about Kaunas")

def generate_response(input_text):
    llm = ChatOpenAI(base_url=endpoint, temperature=0.7, api_key=token, model=model)

    fetched_docs = vectorstore.search(input_text, search_type="similarity", k=3)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    
    result = rag_chain.invoke(input_text)
    st.info(result)

    st.subheader("📚 Sources")
    for i, doc in enumerate(fetched_docs, 1):
        with st.expander(f"Source {i}"):
            source = doc.metadata.get('source', 'Unknown source')
            st.write(f"**Source:** {source}")
            st.write(f"**Content:** {doc.page_content}")

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        placeholder="Type your question here...",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)

