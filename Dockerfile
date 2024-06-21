FROM python:3.9

WORKDIR /app

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

#creating install layers
RUN pip install --upgrade pip setuptools wheel
RUN pip install numpy==1.22.3 pandas==1.4.2 scikit-learn
RUN pip install transformers==4.18.0 tokenizers==0.12.1 soundfile==0.10.3.post1 moviepy
RUN pip install torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

#install extra dependencies in one layer
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ./cache.py ./
RUN python cache.py

# Run data preprocessing (optional if ckpt has been pre trained)
#COPY ./data_preprocessing.sh ./
#RUN chmod +x data_preprocessing.sh
#RUN ./data_preprocessing.sh

# used to download training dependencies (optional if ckpt has been pre trained)
#COPY ./train_crossattention.py ./Distill_knowledge.py ./KD_train_crossattention.py ./
#RUN python train_crossattention.py --model_name multimodal_teacher --epoch 1
#RUN python Distill_knowledge.py --teacher_name multimodal_teacher_epoch0
#RUN python KD_train_crossattention.py --model_name multimodal_student
#RUN python KD_train_crossattention.py --model_name text_student --text_only True
#RUN python KD_train_crossattention.py --model_name audio_student --audio_only True

# Copy the rest of your application code
COPY ./data ./data
COPY ./models ./models
COPY ./*.py ./

EXPOSE 5000

COPY ./ckpt/text_student_epoch67.pt ./ckpt/

# Run your application with Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "inference_server:app"]

