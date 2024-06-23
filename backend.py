import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

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
    if startup_bool is None:
        print("Not initialized startup!")
        raise ValueError("Classes have not been initialized.")
    
    instruction = prompt_template.format(user_question=question, retrieved_document=tmp_doc)
    
    # request on http://34.168.84.98:8000/answer/ with json {"instruction": instruction}. Will return {"instruction": instruction, "answer": answer}.
    response = requests.post("http://34.168.84.98:8000/answer/", 
                             headers={"Content-Type": "application/json"}, 
                             data=json.dumps({"instruction": instruction}))
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


@app.on_event("startup")
async def app_startup():
    global startup_bool
    startup_bool = True
    print("startup_bool initialized:", startup_bool)
