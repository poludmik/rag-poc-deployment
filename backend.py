import json
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
from google.cloud import storage
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
# from sentence_transformers.util import cos_sim
from typing import List
import traceback

prompt_template="""<s>[INST] You are a knowledgeable assistant with access to various documents. Use the provided document to answer the following user question accurately and concisely. If the document does not contain the exact information, provide the best possible answer based on the available content.

User Question: '{user_question}'

Retrieved Document:
'{retrieved_document}'

Answer: [/INST]
"""

app = FastAPI()

tmp_doc = "numerical evaluation, which is one of the goals of this work. Visualization in the Era of Artificial Intelligence expands on the same thoughts while also assessing the LLM capabilities to generate code for 2D and 3D scenes in different programming languages. Text2Analysis is an important paper for this work because it significantly contributed to the ideas of evaluating the agents for table analysis and also provided a large dataset of various types of questions for both statistical and visualization questions. To the date of writing this thesis, the dataset wasnt available on the GitHub page stated in the paper; however, the authors kindly gave me access to the pre-release version of the dataset. The dataset contains 2249 question-answer instances with the correct code. Questions are on 347 different tables. The questions ultimately fall into one of the four categories: Rudimentary Operations, Basic Insights, Forecasting, and Chart Generation. The Rudimentary Operations category contains queries on selecting, filtering, and performing simple aggregation operations on the tabular data. Each Rudimentary Operation query instance also has an accompanying list of operation names that are supposed to be performed, e.g., Pivot/groupby, Aggregation. Basic Insights are more difficult tasks, where the agents should know how to see trends in the data, how to detect outliers, etc. The authors state implementing 7 custom functions to get the result for each query. The Forecasting category is aimed at testing the ability to predict the next samples from the available data. This, however, implies generating longer code that uses other Python modules, e.g., Greykite or Prophet. The Chart Generation question set encompasses queries on visualizing the tabular data. The authors also state ambiguities for every task, if those are present, e.g., Unspecified tasks ambiguity for the query Analyze the data. They put additional effort into making queries more difficult and concentrate on more unclear queries, where the column names are not specified directly or the task could easily be interpreted differently."

def get_answer(filename: str, question: str) -> str:
    """Get answer to question."""
    if not startup_bool:
        print("Not initialized startup!")
        raise ValueError("Classes have not been initialized.")
    
    # Get the index
    get_index(filename)

    folder = filename.split(".")[0]
    db = FAISS.load_local(f"faiss_dbs/indexes/{folder}", GTEEmbeddings().embed_documents, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(k=3)
    docs = retriever.invoke(question)

    # answer = "\n\n".join([doc.page_content for doc in docs])

    instruction = prompt_template.format(user_question=question, retrieved_document=docs[0].page_content)

    # request on http://34.168.84.98:8000/answer/ with json {"instruction": instruction}. Will return {"instruction": instruction, "answer": answer}.
    response = requests.post("http://34.168.84.98:8000/answer/", 
                             headers={"Content-Type": "application/json"}, 
                             data=json.dumps({"instruction": instruction}),
                             timeout=30)

    if response.status_code != 200:
        raise ValueError(f"Request failed with status {response.status_code}: {response.text}")
    response_data = response.json()
    answer = response_data.get("answer", "No answer produced.")

    print("type(answer):", type(answer))

    return answer

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the backend!"}

# Define the request model
class QuestionRequest(BaseModel):
    filename: str
    question: str

# FastAPI endpoint for answering questions
@app.post("/answer/")
async def answer(request: QuestionRequest):
    """Endpoint for answering questions."""
    try:
        print("filename:", request.filename)
        print("question:", request.question)
        answer = get_answer(request.filename, request.question)
        print("answer:", answer)
        return {"filename": request.filename, "question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_pdfs")
def list_pdfs():
    """Lists all the blobs in the bucket."""
    try:
        storage_client = storage.Client()
        pdfs = storage_client.list_blobs("bucket-temus-test-case")
        return {"answer": [blob.name[5:] for blob in pdfs if blob.name.startswith("pdfs/") and blob.name.endswith(".pdf")]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_index(pdf_name: str):
    """Get the index from indexes/ by pdf name."""
    try:
        pdf_name = pdf_name.split(".")[0]

        storage_client = storage.Client()
        bucket = storage_client.bucket("bucket-temus-test-case")
    
        # Create the local directory if it does not exist
        if not os.path.exists("faiss_dbs/"):
            os.makedirs("faiss_dbs/")
        if not os.path.exists("faiss_dbs/indexes/"):
            os.makedirs("faiss_dbs/indexes/")
    
        # List blobs in the specified folder
        blobs = bucket.list_blobs(prefix="indexes/")
    
        for blob in blobs:
            if not pdf_name in blob.name:
                continue
            print("blob.name:", blob.name)
            blob_name = blob.name
            relative_path = os.path.relpath(blob_name, "indexes/")
            local_file_path = os.path.join("faiss_dbs/indexes/", relative_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            blob.download_to_filename(local_file_path)
            print(f"Downloaded {blob_name} to {local_file_path}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class GTEEmbeddings:

    def __init__(self):
        self.model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text.lower().replace('\n', ' ')]).tolist()

@app.post("/create_and_upload", responses={200: {'Response': "File uploaded"}})
async def create_and_upload(file: UploadFile = File(...)):
    """
    Save file locally and upload it to the GC Storage along with the index.
    """
    try:
        if not file.filename or file.filename == '':
            raise HTTPException(status_code=400, detail="No file selected")

        if file.filename and file.filename.endswith('.pdf'):
            print("file.filename:", file.filename)
            os.makedirs('uploading_files/', exist_ok=True)
            file_content = await file.read()
            file_path_local = f"uploading_files/{file.filename}"
            print("file_path_local:", file_path_local)

            with open(file_path_local, 'wb') as f:
                f.write(file_content)

            storage_client = storage.Client()
            bucket = storage_client.bucket("bucket-temus-test-case")
            blob = bucket.blob(f"pdfs/{file.filename}")
            blob.upload_from_filename(file_path_local)
            print(f"File {file.filename} uploaded to bucket.")

            # create an index and upload it to the bucket also
            raw_documents = PyMuPDFLoader(file_path_local).load_and_split()

            texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(raw_documents)

            db = FAISS.from_documents(texts, GTEEmbeddings())
            store_path = f"{file.filename.split('.')[0]}"
            db.save_local(f'faiss_dbs/indexes/{store_path}')
            # upload the index folder with two files to the bucket
            print("store_path:", store_path)
            for file_name in os.listdir(f'faiss_dbs/indexes/{store_path}'):
                print("file_name:", file_name)
                blob = bucket.blob(f"indexes/{store_path}/{file_name}")
                blob.upload_from_filename(f'faiss_dbs/indexes/{store_path}/{file_name}')
                print(f"File {store_path} uploaded to bucket.")

            # remove the local files
            os.remove(f'faiss_dbs/indexes/{store_path}/index.pkl')
            os.remove(f'faiss_dbs/indexes/{store_path}/index.faiss')
            os.rmdir(f'faiss_dbs/indexes/{store_path}')

            os.remove(file_path_local)

            return {'answer': "PDF file was uploaded to GC Storage with id: {}".format(file.filename)}
        else:
            raise HTTPException(status_code=415, detail="Make sure that the file type is PDF.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=traceback.format_exc())



@app.on_event("startup")
async def app_startup():
    global startup_bool
    startup_bool = True
    print("startup_bool initialized:", startup_bool)


# print(get_index("microsoft_report.pdf"))

