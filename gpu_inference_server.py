from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Example inputs:
user_question = "What is the main difference of functions style from the simple?"
retrieved_document = """Moreover, the experimental section of this thesis introduces different evaluation approaches, leading to the development of two distinct code generation
styles: "simple" and "functions". The simple style involves directing a Coder
LLM to generate a code snippet that executes within a 'main' block, without
the need to create a separate function to fulfill the user's request. The text
answer is simply retrieved from print() statements, generated directly by an
LLM. In contrast, the functions style involves prompting an LLM to either
generate or fill in a specific function (i.e., def solve(df: DataFrame)) that
can later be called on a DataFrame object, allowing the execution result
to be conveniently captured in a variable. This method facilitates direct
comparison with a reference output, ensuring accurate and straightforward
1-to-1 evaluation."""


prompt_template="""<s>[INST] You are a knowledgeable assistant with access to various documents. Use the provided document to answer the following user question accurately and concisely. If the document does not contain the exact information, provide the best possible answer based on the available content.

User Question: '{user_question}'

Retrieved Document:
'{retrieved_document}'

Answer: [/INST]
"""


app = FastAPI()

def get_answer(instruction: str, document: str) -> str:
    """Run LLM inference."""
    if llm is None:
        raise ValueError("LLM has not been initialized.")
    
    input_prompt = prompt_template.format(user_question=instruction, retrieved_document=document)
    sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=512)
    outputs = llm.generate([input_prompt], sampling_params)
    answer = outputs[0].outputs[0].text
    return answer

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the LLM backend!"}


class QuestionRequest(BaseModel):
    instruction: str
    document: str

@app.post("/answer/")
async def answer(request: QuestionRequest):
    """Endpoint for generating text."""
    try:
        print("instruction:", request.instruction)
        print("document:", request.document)
        answer = get_answer(request.instruction, request.document)
        print("answer:", answer)
        return {"instruction": request.instruction, "document": request.document, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def app_startup():
    global llm
    llm = LLM(
        model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
        quantization="awq",
        dtype="auto",
        gpu_memory_utilization=0.9,  # Adjust this value as needed
        max_model_len=23000           # Adjust this value to be less than or equal to 23088
    )
