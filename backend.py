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
# import tabula

prompt_template="""You are a knowledgeable assistant with access to documents. Use the provided document to answer the following user question accurately and concisely. If the document does not contain the exact information, provide the best possible answer based on the available content.

User Question: '{user_question}'

Provided Document:
'{retrieved_document}'"""

app = FastAPI()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
STRAT = "small-to-big" # '1', 'all' or 'small-to-big'


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


def combine_docs(docs: List[str], method: str, folder_pdf: str = None) -> str:
    if method == "1":
        return docs[0]
    elif method == "all":
        return "\n\n".join(docs)
    elif method == "small-to-big":
        if not folder_pdf:
            raise ValueError("Folder PDF not provided.")
        print("folder_pdf:", f"faiss_dbs/indexes/{folder_pdf}/{folder_pdf}.pdf")
        raw_documents = PyMuPDFLoader(f"faiss_dbs/indexes/{folder_pdf}/{folder_pdf}.pdf").load_and_split()
        texts = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(raw_documents)
        idx = 1
        for i in range(len(texts) - 1):
            if texts[i].page_content == docs[0]:
                idx = i
                print("Found index:", idx)
                break

        return texts[idx-1].page_content + " " + docs[0] + " " + texts[idx+1].page_content


def get_answer(filename: str, question: str, model: str) -> str:
    """Get answer to question."""
    if not startup_bool:
        print("Not initialized startup!")
        raise ValueError("Classes have not been initialized.")
    
    # Get the index
    get_index(filename)

    print("got index")

    folder = filename.split(".")[0]
    db = FAISS.load_local(f"faiss_dbs/indexes/{folder}", GTEEmbeddings().embed_documents, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(k=3)
    docs = retriever.invoke(question)
    print("Metadata:", docs[0].metadata)
    print("Retrieved documents:", len(docs))

    combined_docs = combine_docs([doc.page_content for doc in docs], STRAT, folder_pdf=folder)

    instruction = prompt_template.format(user_question=question, retrieved_document=combined_docs)

    print("instruction:", instruction)

    if "gpt" in model:
        print("Using GPT model.")
        answer = get_response_from_openai(instruction)
    else:
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

        answer = response.json().get("answer", "No answer produced.")

    delete_files_from_local(f"faiss_dbs/indexes/{folder}")
    
    return answer, combined_docs

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
        print(f"filename: {request.filename}\nquestion: {request.question}\nmodel: {request.model}")
        answer, retrieved_docs = get_answer(request.filename, request.question, request.model)
        print("answer:", answer)
        return {"filename": request.filename, "question": request.question, "answer": answer, "combined_docs": retrieved_docs}
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
        for prefix in ["indexes/", "pdfs/"]:
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if not pdf_name in blob.name:
                    continue
                print("blob.name:", blob.name)
                blob_name = blob.name
                relative_path = os.path.relpath(blob_name, "indexes/")
                print("relative_path:", relative_path)
                local_file_path = os.path.join("faiss_dbs/indexes/", relative_path)
                print("local_file_path:", local_file_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                if prefix == "pdfs/":
                    local_file_path = f"faiss_dbs/indexes/{pdf_name}/{pdf_name}.pdf"
                blob.download_to_filename(local_file_path)
                print(f"Downloaded {blob_name} to {local_file_path}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def delete_files_from_local(dir_name: str):
    """Delete all files from the given directory and the directory itself."""
    try:
        for file_name in os.listdir(dir_name):
            file_path = os.path.join(dir_name, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                delete_files_from_local(file_path)
        os.rmdir(dir_name)
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
            os.makedirs('uploading_files/', exist_ok=True)
            file_content = await file.read()
            file_path_local = f"uploading_files/{file.filename}"

            with open(file_path_local, 'wb') as f:
                f.write(file_content)

            storage_client = storage.Client()
            bucket = storage_client.bucket("bucket-temus-test-case")
            blob = bucket.blob(f"pdfs/{file.filename}")
            blob.upload_from_filename(file_path_local)

            raw_documents = PyMuPDFLoader(file_path_local).load_and_split()

            texts = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(raw_documents)

            # for text in texts:
            #     print(text.metadata)
            # print("len(texts):", len(texts))
            filtered_texts = [text for text in texts if len(text.page_content) >= CHUNK_SIZE-100]
            # max_page_number = max([text.metadata['page'] for text in filtered_texts])

            # parse_tables(file_path_local)

            # dfs = tabula.read_pdf(file_path_local, stream=True, pages="all")

            # for i in range(max_page_number):
            #     # print(filtered_texts[i].page_content)
            #     dfs = tabula.read_pdf(file_path_local, stream=True, pages=str(i+1))
            #     print(dfs)

            db = FAISS.from_documents(filtered_texts, GTEEmbeddings())
            store_path = f"{file.filename.split('.')[0]}"
            db.save_local(f'faiss_dbs/indexes/{store_path}')

            # upload the index folder with two files to the bucket
            for file_name in os.listdir(f'faiss_dbs/indexes/{store_path}'):
                blob = bucket.blob(f"indexes/{store_path}/{file_name}")
                blob.upload_from_filename(f'faiss_dbs/indexes/{store_path}/{file_name}')
                print(f"(2) File {store_path} uploaded to bucket.")

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



