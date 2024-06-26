
# Financial Report Q&A Chatbot (RAG)

## :dart: Project Goals 

The primary objective of this project is to develop a generative Q&A chatbot capable of answering questions related to financial reports of different companies. The chatbot leverages language models and is grounded in a knowledge base constructed from PDF files of financial reports.

### :sparkles: Features

- **Knowledge Base:** Uses PDF files from financial reports to build a robust knowledge base (new PDFs can be uploaded through the UI).
- **Grounding Techniques:** Ensures minimal hallucination by grounding responses in retrieved documents with FAISS indexes and `gte-base-en-v1.5` embeddings.
- **Cloud Deployment:** Utilizes cloud-based SaaS offerings for scalable and robust performance.
- **Dual Model Backend:** Integrates both `GPT-3.5-Turbo` from OpenAI and `Mistral-7B-Instruct-v0.2-AWQ` for model inference.

### :wrench: Technologies Used

- **Backend:** FastAPI for serving the backend API.
- **Frontend:** Streamlit for an interactive user interface.
- **Cloud Services:** Google Cloud Platform (GCP) for continuous deployment and storage.
- **Language Models:** `GPT-3.5-Turbo` and `Mistral-7B-Instruct-v0.2-AWQ` for inference and `gte-base-en-v1.5` for embeddings.

### :file_folder: Project Structure

- **backend.py:** Main backend server handling API requests.
- **gpu_inference_server.py:** Server running the Mistral LLM for GPU-based inference.
- **frontend.py:** Streamlit-based frontend application.
- **utils.py:** Utility functions for embeddings and OpenAI API calls.
- **cloudbuild.yaml:** Configuration for building and deploying Docker images on GCP automatically.
- **dockerfiles/:** Dockerfiles for backend and frontend containers.
- **scripts/:** Example scripts for managing GCP instances and starting the LLM server.


#### Requirements and technologies used
Requirements are separate for backend, frontend and llm inference server. They are stored in the **requirements** folder. The main technologies that were used are:
- Docker
- Google Cloud Platform (storage, build/run, compute engine, secret manager)
- Python 3.10
- Streamlit
- FastAPI
- Huggingface transformers
- OpenAI API

### :computer: Backend Server

The backend server is built using FastAPI and handles the main logic for processing questions, retrieving documents, and calling the appropriate LLM for inference.

#### Key Functions

- **combine_docs:** Combines retrieved documents based on different strategies for improved context. Either the best vector similarity, top-k similarities docs, or Small-to-Big doc expansion.
- **get_answer:** Retrieves documents from the index, combines them, and generates an answer using the selected LLM.
- **get_index**: Retrieves the index for a given filename from the Google Cloud Storage bucket.

#### Supported endpoints
- **answer**: POST request to `/answer/` with a JSON payload containing the filename, question, and model to use for inference.

```python
@app.post("/answer/")
async def answer(request: QuestionRequest):
    try:
        answer, retrieved_docs = get_answer(request.filename, request.question, request.model)
        return {"filename": request.filename, "question": request.question, "answer": answer, "combined_docs": retrieved_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- **list_pdfs**: GET request to `/list_pdfs/` to list all uploaded PDF files that are currently in the GC bucket.
- **create_and_upload**: POST request to `/create_and_upload/` to upload a new PDF file for indexing and querying.

### :zap: GPU Inference Server

The GPU inference server is used to run the Mistral-7B-Instruct-v0.2-AWQ model for generating responses. It utilizes `vllm` for efficient model management and inference. Since the model is large and requires a GPU for inference, a separate server is used to offload the computation from the main backend server. This also allows for better resource management and scalability.

#### Starting the Server

The following simple command is used to start the GPU inference server inside the Google Cloud Compute Engine instance. The server is started in the background and continues to run even when the SSH session is closed.
```sh
nohup uvicorn --host 0.0.0.0 --port 8000 gpu_inference_server:app &
```

### :art: Frontend

The frontend is built with Streamlit and provides an interactive interface for users to upload and select PDFs, ask questions, and view answers.

#### User Options

- **Select Existing PDF:** Choose from already uploaded PDFs.
- **Upload New PDF:** Upload a new PDF file for indexing and querying.
- **Select LLM Model:** Choose between `GPT-3.5-Turbo` and `Mistral-7B-Instruct-v0.2-AWQ` for inference.
- **Ask Questions:** Input questions related to the selected PDF and get responses.

```python
if question := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.write(question)
    with st.spinner('Please wait...'):
        response = requests.post(answer_url, json.dumps({
            'question': question,
            'filename': st.session_state.current_file_name,
            'model': "mistral" if on else "gpt"
        }))
        st.write(response.json()["answer"])
```

### :cloud: Cloud Build Configuration

The `cloudbuild.yaml` file automates the process of building Docker images and deploying them to Google Cloud Run. The backend and frontend are deployed as separate services. The configuration file specifies the steps to build the Docker images, tag them with the appropriate version, and deploy them to Cloud Run. The Cloud Build **trigger** for this build is set to automatically deploy the latest changes to the main branch.

#### Demo Github Actions Workflow
A dummy test is set in the .github/workflows folder. The workflow is triggered on push or pull request to the main branch. The workflow runs a test that always passes. This is just a placeholder for the actual tests that could be implemented in the future.
