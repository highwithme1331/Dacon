#Basic Web Page Data
def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return re.sub(r'\s{2,}', ' ', ' '.join(non_empty)).strip()


from langchain.document_loaders import WebBaseLoader

basic_loader = WebBaseLoader(news_url)

try:
    basic_docs = basic_loader.load()

    for doc in basic_docs:
        doc.page_content = clean_text(doc.page_content)

except Exception as e:
    print("Error")



#Paragraph Web Page Data
content_loader = WebBaseLoader(
    news_url,
    bs_kwargs={"parse_only": bs4.SoupStrainer("p")}
)

try:
    content_docs = content_loader.load()

    for doc in content_docs:
        doc.page_content = clean_text(doc.page_content)

except Exception as e:
    print("Error")



#Multiple Web Page Data
batch_loader = WebBaseLoader(
    news_urls,
    bs_kwargs={"parse_only": bs4.SoupStrainer("p")},
    requests_kwargs={
        "headers": {"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"},
        "timeout": 15,
        "verify": True
    }
)

try:
    batch_docs = batch_loader.load()

    for doc in batch_docs:
        doc.page_content = clean_text(doc.page_content)

except Exception as e:
    print("Error")



#Metadata Web Page Data
seoul_tz = ZoneInfo("Asia/Seoul")
now = datetime.now(seoul_tz)
collection_time = now.strftime("%Y년 %m월 %d일 %H시 %M분 %S초")

enhanced_docs = []

for doc in batch_docs:
    enhanced_meta = {
        **doc.metadata,
        "collection_time": collection_time,
        "document_type": "news_article",
        "content_length": len(doc.page_content),
        "word_count": len(doc.page_content.split())
    }

    enhanced_docs.append(Document(page_content=doc.page_content, metadata=enhanced_meta))



#PDF
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader

pdf_path = "pdf"

loader_a = PyPDFLoader(pdf_path)
docs_a = loader_a.load()

loader_b = PDFPlumberLoader(pdf_path)
docs_b = loader_b.load()



#Basic Split
from langchain_text_splitters import RecursiveCharacterTextSplitter

basic_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

structure_aware_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n# ", "\n## ", "\n\n", "\n- ", "\n", ". ", " ", ""]
)

basic_chunks = basic_splitter.split_text(sample_document)
smart_chunks = structure_aware_splitter.split_text(sample_document)



#Token Split
from langchain_text_splitters import TokenTextSplitter

token_splitter = TokenTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    encoding_name="cl100k_base"
)

token_chunks = token_splitter.split_text(sample_document)



#Semantic Split
from langchain_experimental.text_splitter import SemanticChunker

semantic_splitter = SemanticChunker(
    embeddings, 
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=70
)

semantic_chunks = semantic_splitter.split_text(sample_document)



#Embedding
from langchain_openai import OpenAIEmbeddings

emb_docs_small = emb_small.embed_documents(docs)
emb_docs_large = emb_large.embed_documents(docs)

q_small = emb_small.embed_query(query)
q_large = emb_large.embed_query(query)

def cosine_similarity(a, b):
    a_np, b_np = np.array(a), np.array(b)
    
    return float(a_np.dot(b_np)/(np.linalg.norm(a_np)*np.linalg.norm(b_np)))

def top_k(query_vec, doc_vecs, docs, k=3):
    sims = [cosine_similarity(query_vec, dv) for dv in doc_vecs]
    idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    
    return [(i, docs[i], sims[i]) for i in idxs]

top_small = top_k(q_small, emb_docs_small, docs, k=3)
top_large = top_k(q_large, emb_docs_large, docs, k=3)