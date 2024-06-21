from fastapi import FastAPI
from http import HTTPStatus

app = FastAPI()


@app.get("/")
def root():
    """ Check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/answer")
def read_item(question: str):
    return {"question_was": question, "answer": "228"}
