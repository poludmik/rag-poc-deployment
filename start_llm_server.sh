cd temus-case-study
nohup uvicorn --host 0.0.0.0 --port 8000 gpu_inference_server:app &

# make executable:
# chmod +x start_llm_server.sh

# to end:
# ps -ef | grep uvicorn 
# kill <pid>