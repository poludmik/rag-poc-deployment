import os
from fastapi import FastAPI, HTTPException, UploadFile, File
import openai
from sentence_transformers import SentenceTransformer
from typing import List
import openai
from openai import OpenAI


class GTEEmbeddings:
    """
    Embeddings that are used in indices for RAG vector search.
    """
    def __init__(self):
        self.model = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
        # self.model = SentenceTransformer('Mihaiii/gte-micro', trust_remote_code=True)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text.lower().replace('\n', ' ')]).tolist()
    

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


def get_response_from_openai(prompt: str) -> str:
    """Get response from OpenAI API."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found.")
    client = OpenAI()
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=[
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content


prompt_template="""You are a knowledgeable assistant with access to documents. Use the provided document to answer the following user question accurately and concisely. If the document does not contain the exact information, provide the best possible answer based on the available content.

User Question: '{user_question}'

Provided Document:
'{retrieved_document}'"""
