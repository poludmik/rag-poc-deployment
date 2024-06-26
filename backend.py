import json
import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import requests
from google.cloud import storage
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import List
import traceback
# import tabula
import logging
import sys
from rich.logging import RichHandler
from utils import *
import shutil


app = FastAPI()

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.root.handlers[0] = RichHandler(markup=True)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 0
STRAT = "small-to-big" # '1' - first best chunk, 'all' - top-k chunks, 'small-to-big' - the top chunk expanded with 2 surrounding chunks


def combine_docs(docs: List[str], method: str, folder_pdf: str = None) -> str:
    if method == "1":
        return docs[0]
    elif method == "all":
        return "\n\n".join(docs)
    elif method == "small-to-big":
        if not folder_pdf:
            raise ValueError("Folder PDF not provided.")
        logger.debug("folder_pdf: %s", f"faiss_dbs/indexes/{folder_pdf}/{folder_pdf}.pdf")
        raw_documents = PyMuPDFLoader(f"faiss_dbs/indexes/{folder_pdf}/{folder_pdf}.pdf").load_and_split()
        texts = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(raw_documents)
        idx = 1
        for i in range(len(texts) - 1):
            if texts[i].page_content == docs[0]:
                idx = i
                logger.debug("Found index: %s", idx)
                break
        return texts[idx-1].page_content + " " + docs[0] + " " + texts[idx+1].page_content


def get_answer(filename: str, question: str, model: str) -> str:
    """
    Get answer to question.
    Download index from GC Storage, retrieve documents, combine them, ask LLM, and return answer.
    """
    get_index(filename)

    logger.debug("Got index")

    folder = filename.split(".")[0]
    db = FAISS.load_local(f"faiss_dbs/indexes/{folder}", GTEEmbeddings().embed_documents, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(k=3)
    docs = retriever.invoke(question)
    logger.debug("Metadata: %s, Retrieved %s documents.", docs[0].metadata, len(docs))

    combined_docs = combine_docs([doc.page_content for doc in docs], STRAT, folder_pdf=folder)

    page_number = docs[0].metadata['page'] + 1

    instruction = prompt_template.format(user_question=question, retrieved_document=combined_docs)
    logger.debug("instruction: %s", instruction)

    if "gpt" in model:
        logger.debug("Using GPT model.")
        answer = get_response_from_openai(instruction)
    else:
        instruction = "<s>[INST] " + instruction + "\n\nAnswer: [/INST]\n"
        logger.debug("Using Mistral-7B model.")
        response = requests.post("http://34.83.196.140:8000/answer/", 
                                 headers={"Content-Type": "application/json"}, 
                                 data=json.dumps({"instruction": instruction}),
                                 timeout=15)

        if response.status_code != 200:
            logger.warning("First request failed with status %s.", response.status_code)
            response = requests.post("http://34.168.84.98:8000/answer/", # trying another compute instance
                                        headers={"Content-Type": "application/json"},
                                        data=json.dumps({"instruction": instruction}),
                                        timeout=15)
            if response.status_code != 200:
                logger.error("Second request failed with status %s.", response.status_code)
                raise ValueError(f"Request failed with status {response.status_code}: {response.text}")

        answer = response.json().get("answer", "No answer produced.")

    delete_files_from_local(f"faiss_dbs/indexes/{folder}")
    
    return answer, combined_docs + f"\n\n**Retrieved from around page {page_number}**" if STRAT == "small-to-big" or STRAT == "1" else combined_docs


@app.get("/")
async def root():
    return {"message": "Hello from the backend!"}


class QuestionRequest(BaseModel):
    filename: str
    question: str
    model: str


@app.post("/answer/")
async def answer(request: QuestionRequest):
    """Endpoint for answering questions."""
    try:
        logger.debug(f"filename: {request.filename}\nquestion: {request.question}\nmodel: {request.model}")
        answer, retrieved_docs = get_answer(request.filename, request.question, request.model)
        logger.debug("answer: %s", answer)
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
    
        if not os.path.exists("faiss_dbs/"):
            os.makedirs("faiss_dbs/")
        if not os.path.exists("faiss_dbs/indexes/"):
            os.makedirs("faiss_dbs/indexes/")
    
        for prefix in ["indexes/", "pdfs/"]:
            blobs = bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                if not pdf_name in blob.name:
                    continue
                logger.debug("blob.name: %s", blob.name)
                blob_name = blob.name
                relative_path = os.path.relpath(blob_name, "indexes/")
                logger.debug("relative_path: %s", relative_path)
                local_file_path = os.path.join("faiss_dbs/indexes/", relative_path)
                logger.debug("local_file_path: %s", local_file_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                if prefix == "pdfs/":
                    local_file_path = f"faiss_dbs/indexes/{pdf_name}/{pdf_name}.pdf"
                blob.download_to_filename(local_file_path)
                logger.info(f"Downloaded {blob_name} to {local_file_path}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_and_upload", responses={200: {'Response': "File uploaded"}})
async def create_and_upload(file: UploadFile = File(...)):
    """
    Save the pdf file locally and upload it to the GC Storage along with the index.
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

            filtered_texts = [text for text in texts if len(text.page_content) >= CHUNK_SIZE-100]

            # print(texts[0].metadata)
            # max_page_number = max([text.metadata['page'] for text in filtered_texts])
            # dfs = tabula.read_pdf(file_path_local, stream=True, pages="all")

            db = FAISS.from_documents(filtered_texts, GTEEmbeddings())
            store_path = f"{file.filename.split('.')[0]}"
            db.save_local(f'faiss_dbs/indexes/{store_path}')

            for file_name in os.listdir(f'faiss_dbs/indexes/{store_path}'):
                blob = bucket.blob(f"indexes/{store_path}/{file_name}")
                blob.upload_from_filename(f'faiss_dbs/indexes/{store_path}/{file_name}')
                logger.info(f"File {file_name} uploaded to bucket.")

            shutil.rmtree(f'faiss_dbs/indexes/{store_path}', ignore_errors=True)

            os.remove(file_path_local)

            return {'answer': "PDF file was uploaded to GC Storage with id: {}".format(file.filename)}
        else:
            raise HTTPException(status_code=415, detail="Make sure that the file type is PDF.")

    except Exception as e:
        raise HTTPException(status_code=400, detail=traceback.format_exc())
