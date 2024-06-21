FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    curl \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#creating install layers
RUN pip install --upgrade pip
#install extra dependencies in one layer
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./cache.py ./
RUN python cache.py

# Copy the rest of your application code
COPY ./models ./models
COPY ./*.py ./

EXPOSE 5000

COPY ./ckpt/text_student_epoch67.pt ./ckpt/

# Run your application with Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "inference_server:app"]

