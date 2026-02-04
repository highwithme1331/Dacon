#Vector Store Filtering Search
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client()

collection = client.create_collection(
    name="collection",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=[f"doc_{i}" for i in range(len(documents))]
)


basic_results = collection.query(
    query_texts=["query"],
    n_results=3
)


filtered_results = collection.query(
    query_texts=["query"],
    n_results=3,
    where={
        "$and": [
            {"A": {"$eq": "category_A"}},
            {"B": {"$eq": 1000}},
            {"C": {"$eq": "category_C"}}
        ]
    }
)


complex_results = collection.query(
    query_texts=["query"],
    n_results=5,
    where={
        "$and": [
            {"A": {"$eq": "category_A"}},
            {"B": {"$gte": 1000}},
            {"C": {"$in": ["category_C1", "category_C2"]}}
        ]
    }
)



#Retriever
from langchain.vectorstores import Chroma
from langchain.retrievers import MultiQueryRetriever, EnsembleRetriever, BM25Retriever

documents = [Document(page_content=doc) for doc in tech_documents]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="collection"
)


basic_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

basic_results = basic_retriever.invoke(query)


multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=basic_retriever,
    llm=llm
)

multi_results = multi_query_retriever.invoke(query)


bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 3


ensemble_retriever = EnsembleRetriever(
    retrievers=[basic_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

ensemble_results = ensemble_retriever.invoke(query)