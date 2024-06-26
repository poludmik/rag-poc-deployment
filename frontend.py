import os
from io import StringIO
import streamlit as st
import requests
import json
import base64
from google.cloud import run_v2

def get_backend_url():
    """Get the URL of the backend service automatically."""
    parent = "projects/temusrag/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "backend":
            return service.uri
    name = os.environ.get("BACKEND", None)
    return name

url = get_backend_url()
# url = "http://127.0.0.1:8000/"

upload_url = url + '/create_and_upload'
answer_url = url + '/answer/'
list_url = url + '/list_pdfs'

st.set_page_config(layout="wide")
st.write("""# Temus RAG demo""")

dict_with_available = json.loads(requests.get(list_url).text)
list_with_available_pdfs = [s for s in dict_with_available["answer"]]

selectbox = st.sidebar.selectbox(
    "Select an already uploaded PDF file:",
    list_with_available_pdfs
)
if 'selectbox_value' not in st.session_state:
    st.session_state.selectbox_value = selectbox
    st.df = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = list_with_available_pdfs[0] if len(list_with_available_pdfs) > 0 else None


file = st.sidebar.file_uploader("Or upload a new file:", type=["pdf"])
if file is not None:
    if st.sidebar.button("Upload file"):
        with st.spinner('File is being indexed and uploaded...'):
            resp = requests.post(upload_url, files={'file': file})
            if resp.status_code == 200:
                st.write("File uploaded succesfully")
                st.session_state.current_file_name = file.name
                st.session_state.selectbox_value = file.name
                print("(Upload) Changed file to:", st.session_state.current_file_name)
            else:
                print(resp.text)
                st.write("An error occured during file upload!")

available_models = ["Mistral-7B-AWQ (may be off)", "gpt-3.5-turbo-0125"]
on = st.sidebar.toggle("LLM selection", False)

if on:
    st.sidebar.write("**Mistral-7B-AWQ** is now selected. The GPU server may be off.")
else:
    st.sidebar.write("**gpt-3.5-turbo-0125** is now selected.")


if selectbox != st.session_state.selectbox_value:
    st.session_state.selectbox_value = selectbox
    
    if file is not None:
        current_filename = file.name
        print("file.name:", current_filename)
    else:
        current_filename = selectbox
    st.session_state.current_file_name = current_filename
    print("(Selectbox) Changed file to:", st.session_state.current_file_name)
    filename = st.session_state.current_file_name
    with st.chat_message("ai"):
        st.write("Please ask a question on **" + st.session_state.current_file_name + "**:")
else:
    pass

if question := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.write(question)
    with st.spinner('Please wait...'):
        print("question was:", question)
        response = requests.post(answer_url, json.dumps({'question': question, 'filename': st.session_state.current_file_name, 'model': "mistral" if on else "gpt"}))
        print("response:", response, "type:", type(response))
        print("RESPONSE CONTENT:", json.loads(response.content.decode()))
        with st.chat_message("ai"):
            st.write("Here is the answer to your question along with the retrieved context:")
            try:
                st.write("### gpt-3.5-turbo-0125 answer:" if not on else "### Mistral-7B-AWQ answer:")
                st.write(json.loads(response.content.decode())["answer"].replace("\n", "  \n")) # Streamlit write only recognizes 'word  \n' and not 'word\n'... (Uses markdown)
                st.write("### Retrieved context:")
                st.write(json.loads(response.content.decode())["combined_docs"].replace("\n", "  \n"))
            except Exception as e:
                st.write("An error occured during answering the question. " + "Perhaps the GPU compute instance is off?" if on else "")
