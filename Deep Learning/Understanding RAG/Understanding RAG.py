#Document Loaders
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdf")
documents = loader.load()



#Text Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=300,
    chunk_overlap=20,
)

chunks = text_splitter.split_documents(documents)



#Embeddings
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key="발급 받은 API key"
)

embedded_text = embeddings.embed_query(text)



#Vector Stores
from langchain_community.vectorstores import FAISS

db_faiss = FAISS.from_documents(documents=chunks, embedding=embeddings)
docs_faiss = db_faiss.similarity_search(query, k=2)



#Retriever
retriever = vectorstore.as_retriever(
   search_type="similarity",
   search_kwargs={"k": 2}
)



#Prompt Template
from langchain.prompts import PromptTemplate

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)



#LLM Load
from langchain_openai import ChatOpenAI

openai_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0.01, 
    openai_api_key='발급 받은 API key'
)



#RAG Chain
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=openai_llm,
    chain_type="stuff",
    retriever= retriever,
    chain_type_kwargs={"prompt": PROMPT}, 
    return_source_documents=True
)