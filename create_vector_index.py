from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
# from sentence_transformers.util import cos_sim
from typing import List


# texts = ['That is a happy person', 'That is a very happy person']
# print(cos_sim(embeddings[0], embeddings[1]))


class GTEEmbeddings:

    def __init__(self):
        self.model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text.lower().replace('\n', ' ')]).tolist()


# raw_documents = PyMuPDFLoader('data/microsoft_report.pdf').load_and_split()

# texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(raw_documents)

# db = FAISS.from_documents(texts, GTEEmbeddings())
# store_path = 'microsoft_report_gte_faiss_db'
# db.save_local('./faiss_dbs/indexes/' + store_path)
# print(f"Database stored in {store_path} folder")

# print("len(texts):", len(texts))
# print(type(texts[12]))


# print(texts[12].page_content)


# embed query
# query = 'That is a happy person'
# query_embedding = GTEEmbeddings().embed_query(query)
# print(len(query_embedding[0]))

db = FAISS.load_local("faiss_dbs/indexes/microsoft_report_gte_faiss_db", GTEEmbeddings().embed_documents, allow_dangerous_deserialization=True)

retriever = db.as_retriever(k=4)

query = """We provide for the estimated costs of fulfilling our obligations under hardware and software warranties at the time the related
revenue is recognized. For hardware warranties, we estimate the costs based on historical and projected product failure
rates, historical and projected repair costs, and knowledge of specific product failures (if any). The specific hardware
warranty terms and conditions vary depending upon the product sold and the country in which we do business, but generally
include parts and labor over a period generally ranging from 90 days to three years. For software warranties, we estimate
the costs to provide bug fixes, such as security patches, over the estimated life of the software. We regularly reevaluate our
estimates to assess the adequacy of the recorded warranty liabilities and adjust the amounts as necessary. """
docs = retriever.invoke(query)

print("len(docs):", len(docs))

print(docs[0].page_content)
print(docs[1].page_content)
