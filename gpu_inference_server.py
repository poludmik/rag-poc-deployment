from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

def get_answer(instruction: str) -> str:
    """Run LLM inference."""
    if llm is None:
        raise ValueError("LLM has not been initialized.")
    
    sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=512)
    outputs = llm.generate([instruction], sampling_params)
    answer = outputs[0].outputs[0].text
    return answer

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the LLM backend!"}


class QuestionRequest(BaseModel):
    instruction: str

@app.post("/answer/")
async def answer(request: QuestionRequest):
    """Endpoint for generating text."""
    try:
        print("instruction:", request.instruction)
        answer = get_answer(request.instruction)
        print("answer:", answer)
        return {"instruction": request.instruction, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def app_startup():
    global llm
    llm = LLM(
        model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        quantization="awq",
        dtype="auto",
        gpu_memory_utilization=0.9,
        max_model_len=23000
    )
