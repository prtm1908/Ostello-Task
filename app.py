from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import torch
from model import load_model, generate_response


app = FastAPI()


# Load your model, tokenizer, and generation config
model, tokenizer, generation_config = load_model()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class QuestionRequest(BaseModel):
    question: str


@app.post("/generate/", response_class=HTMLResponse)
async def generate(question_request: QuestionRequest):
    question = question_request.question
    response = generate_response(model, tokenizer, generation_config, question)
    print(response)
    return f"<h2>{response}</h2>"


@app.get("/")
async def main():
    content = """
    <html>
    <head>
        <title>Ostello Task</title>
    </head>
    <body>
        <h1>Ostello E-Commerce QnA</h1>
        <textarea id="question" rows="4" cols="50"></textarea><br>
        <button onclick="submitQuestion()">Submit</button>
        <div id="responseContainer"></div>

        <script>
            async function submitQuestion() {
                const question = document.getElementById('question').value;
                const response = await fetch('/generate/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ "question": question }),
                });
                const responseData = await response.text();
                document.getElementById('responseContainer').innerHTML = responseData;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)