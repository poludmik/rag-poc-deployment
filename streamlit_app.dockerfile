FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the image
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Set environment variables (if the automatic detection doesn't work later)
# ENV BACKEND="https://backend-c4xxjrjd3q-ew.a.run.app"

# Expose the port
EXPOSE 8080

# Set the entry point
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
