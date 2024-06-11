FROM python:3.9

RUN apt-get update && apt-get install -y \
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
    ffmpeg

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    numpy==1.22.3 \
    pandas==1.4.2 \
    scikit-learn

RUN pip install \
    transformers==4.18.0 \
    tokenizers==0.12.1 \
    soundfile==0.10.3.post1 \
    moviepy

RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu113


# Copy the rest of your application code
COPY ./ckpt ./ckpt
COPY ./data ./data
COPY ./models ./models
COPY ./*.py ./

RUN python cache.py

# Run your application
#CMD ["python", "test_text_emotion_recognition.py"]

