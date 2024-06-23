gcloud compute instances create llm-inference-vm \
    --zone=us-west1-b \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy TERMINATE \
    --address=static-ip-for-llm \
    --tags=http-server,https-server
        # --machine-type=g2-standard-4
    



# to run the script:
# chmod +x run_instance.sh
# ./run_instance.sh
