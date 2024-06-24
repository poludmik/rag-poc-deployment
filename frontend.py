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
# url = "https://backend-c4xxjrjd3q-ew.a.run.app"

upload_url = url + '/create_and_upload'
answer_url = url + '/answer/'
list_url = url + '/list_pdfs'
get_table_url = url + '/get_table'

st.set_page_config(layout="wide")
st.write("""
# Temus RAG demo
""")

dict_with_available = json.loads(requests.get(list_url).text)
dict_with_available["answer"] = [s for s in dict_with_available["answer"]]
# dict_with_available = {"answer": ["McDonalds.pdf", "Apple.pdf"]}
selectbox = st.sidebar.selectbox(
    "Select an already uploaded PDF file:",
    dict_with_available["answer"]
)
if 'selectbox_value' not in st.session_state:
    st.session_state.selectbox_value = selectbox
    st.df = None
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = dict_with_available["answer"][0] if len(dict_with_available["answer"]) > 0 else None


file = st.sidebar.file_uploader("Or upload a new file:", type=["pdf"])
if file is not None:
    if st.sidebar.button("Upload file"):
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
    st.sidebar.write("**gp-3.5-turbo-0125** is now selected.")


if selectbox != st.session_state.selectbox_value:
    # Update session state with new selectbox value
    st.session_state.selectbox_value = selectbox
    
    if file is not None:
        current_filename = file.name
        print("file.name:", current_filename)
    else:
        current_filename = selectbox
    st.session_state.current_file_name = current_filename
    print("(Selectbox) Changed file to:", st.session_state.current_file_name)
    # json_df = json.loads(requests.post(get_table_url, json.dumps({'table_name': st.session_state.current_file_name})).content.decode())["df"]
    # df = pd.read_json(StringIO(json_df))
    # st.df = df
    filename = st.session_state.current_file_name
    st.write("Ask a question on **" + st.session_state.current_file_name + "**:")
    # st.write(df)
# elif st.df is not None:
    # st.write(st.df)
else:
    pass


if question := st.chat_input("What is up?"):
    with st.spinner('Please wait...'):
        print("question was:", question)
        response = requests.post(answer_url, json.dumps({'question': question, 'filename': st.session_state.current_file_name, 'model': "mistral" if on else "gpt"}))
        print("response:", response, "type:", type(response))
    # if json.loads(response.content.decode())["is_image"] == True:
    #     image_data = base64.b64decode(json.loads(response.content.decode())["image_str"])
    #     img = st.image(image_data)
        print("RESPONSE CONTENT:", json.loads(response.content.decode()))
        try:
            st.write("**gp-3.5-turbo-0125 answer:**" if not on else "**Mistral-7B-AWQ answer:**")
            st.write(json.loads(response.content.decode())["answer"].replace("\n", "  \n")) # Streamlit write only recognizes 'word  \n' and not 'word\n'... (Uses markdown)
        except Exception as e:
            st.write("An error occured during answering the question.")
            raise e
    # st.markdown(question)
    # st.text(json.loads(response.content.decode())["answer"]) # Better for printing dataframes
    # st.text("This is a test answer.")
