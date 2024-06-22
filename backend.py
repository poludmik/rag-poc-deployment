from contextlib import asynccontextmanager
import random

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# classes = ['class1', 'class2', 'class3']
# print("Classes:", classes)

from llama_cpp import Llama


def generate_text(
    prompt="Who is the CEO of Apple?",
    max_tokens=256,
    temperature=0.1,
    top_p=0.5,
    echo=False,
    stop=["#"],
):
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        echo=echo,
        stop=stop,
    )
    output_text = output["choices"][0]["text"].strip()
    return output_text


def generate_prompt_from_template(input):
    chat_prompt_template = f"""<|im_start|>system
You are a helpful chatbot.<|im_end|>
<|im_start|>user
{input}<|im_end|>"""
    return chat_prompt_template


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

        llm_answer = generate_text(
            request.question,
            max_tokens=356,
            )

        answer += llm_answer
        print("answer:", answer)
        return {"filename": request.filename, "question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def app_startup():
    global classes
    classes = ['class1', 'class2', 'class3']
    print("Classes initialized:", classes)
    global llm
    llm = Llama(model_path="zephyr-7b-beta.Q4_0.gguf", n_ctx=512, n_batch=126)