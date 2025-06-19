from fastapi import FastAPI
from pydantic import BaseModel
from app.llm_client import ask_gpt

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_question(query: Query):
    answer = ask_gpt(query.question)
    return {"answer": answer}

