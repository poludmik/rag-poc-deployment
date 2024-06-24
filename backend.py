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
import openai
from openai import OpenAI

prompt_template="""You are a knowledgeable assistant with access to various documents. Use the provided document to answer the following user question accurately and concisely. If the document does not contain the exact information, provide the best possible answer based on the available content.

User Question: '{user_question}'

Retrieved Document:
'{retrieved_document}'"""

app = FastAPI()


def get_response_from_openai(prompt: str) -> str:
    """Get response from OpenAI API."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found.")
    print("openai.api_key set")

    client = OpenAI()

    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content


def get_answer(filename: str, question: str, model: str) -> str:
    """Get answer to question."""
    if not startup_bool:
        print("Not initialized startup!")
        raise ValueError("Classes have not been initialized.")
    
    # Get the index
    get_index(filename)

    print("got index")

    folder = filename.split(".")[0]
    print("folder:", folder)
    db = FAISS.load_local(f"faiss_dbs/indexes/{folder}", GTEEmbeddings().embed_documents, allow_dangerous_deserialization=True)
    print("db:", db)
    retriever = db.as_retriever(k=3)
    print("retriever:", retriever)
    docs = retriever.invoke(question)
    print("Retrieved documents:", len(docs))

    # answer = "\n\n".join([doc.page_content for doc in docs])
    instruction = prompt_template.format(user_question=question, retrieved_document=docs[0].page_content)

    print("instruction:", instruction)

    if "gpt" in model:
        print("Using GPT model.")
        return get_response_from_openai(instruction)

    instruction = "<s>[INST] " + instruction + "\n\nAnswer: [/INST]\n"

    print("Using Mistral-7B model.")
    response = requests.post("http://34.83.196.140:8000/answer/", 
                             headers={"Content-Type": "application/json"}, 
                             data=json.dumps({"instruction": instruction}),
                             timeout=15)

    if response.status_code != 200:
        print("FIRST response.status_code:", response.status_code)
        response = requests.post("http://34.168.84.98:8000/answer/", # trying another compute instance
                                    headers={"Content-Type": "application/json"},
                                    data=json.dumps({"instruction": instruction}),
                                    timeout=15)
        print("SECOND response.status_code:", response.status_code)
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
    model: str

# FastAPI endpoint for answering questions
@app.post("/answer/")
async def answer(request: QuestionRequest):
    """Endpoint for answering questions."""
    try:
        print("filename:", request.filename)
        print("question:", request.question)
        print("model:", request.model)
        answer = get_answer(request.filename, request.question, request.model)
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
        # self.model = SentenceTransformer('Mihaiii/gte-micro', trust_remote_code=True)
        
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
            print(f"(1) File {file.filename} uploaded to bucket.")

            raw_documents = PyMuPDFLoader(file_path_local).load_and_split()
            print("len(raw_documents):", len(raw_documents))

            texts = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(raw_documents)
            # print("len(texts):", len(texts))
            # filtered_texts = [text for text in texts if len(text.page_content) >= 400]
            print("len(texts):", len(texts))

            db = FAISS.from_documents(texts, GTEEmbeddings())
            print("db:", db)
            store_path = f"{file.filename.split('.')[0]}"
            print("store_path:", store_path)
            db.save_local(f'faiss_dbs/indexes/{store_path}')
            print("Index saved locally.")
            # upload the index folder with two files to the bucket
            for file_name in os.listdir(f'faiss_dbs/indexes/{store_path}'):
                print("file_name:", file_name)
                blob = bucket.blob(f"indexes/{store_path}/{file_name}")
                print("blob")
                blob.upload_from_filename(f'faiss_dbs/indexes/{store_path}/{file_name}')
                print(f"(2) File {store_path} uploaded to bucket.")

            print("After for loop.")

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

