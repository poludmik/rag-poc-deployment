gcloud compute instances create llm-inference-vm-2cpu \
    --zone=us-west1-b \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy TERMINATE \
    --address=static-ip-for-llm-2cpu \
    --tags=http-server,https-server \
    --machine-type=n1-standard-2


gcloud compute instances create llm-inference-vm-l4 \
    --zone=us-west1-b \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --accelerator="type=nvidia-l4,count=1" \
    --maintenance-policy TERMINATE \
    --address=static-ip-for-llm \
    --tags=http-server,https-server \
    --machine-type=g2-standard-4