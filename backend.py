from contextlib import asynccontextmanager
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# classes = ['class1', 'class2', 'class3']
# print("Classes:", classes)

app = FastAPI()

def get_answer(filename: str, question: str) -> str:
    """Get answer to question."""
    if classes is None:
        print("Classes have not been initialized.")
        raise ValueError("Classes have not been initialized.")
    return f"Answer to '{question}' is {random.choice(classes)}."

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


@app.on_event("startup")
async def app_startup():
    global classes
    classes = ['class1', 'class2', 'class3']
    print("Classes initialized:", classes)
