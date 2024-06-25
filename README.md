# Multi-Still: A Lightweight Multi-modal Cross Attention Knowledge Distillation Method for Real-time Emotion Recognition

## 2nd ETRI Human Understanding AI Paper Competition (2023)
### This competition is hosted by the Electronics and Telecommunications Research Institute (ETRI) and sponsored by the Ministry of Science and ICT and the National Research Council of Science and Technology (NST).

> 😊 Introduction
+ The competition aims to implement AI that can interact with humans.
+ It seeks to expand research on AI technologies that understand human behavior and emotions by utilizing a dataset built for this purpose.
+ The goal is to discover creative research.

> 😊 Host/Sponsor
+ Host: Electronics and Telecommunications Research Institute (ETRI)
+ Sponsors: Ministry of Science and ICT, National Research Council of Science and Technology (NST)
+ Operator: AI Factory (AIFactory)

> 😊 Paper Topic
+ Emotion recognition technology using a multi-modal emotion dataset.
+ Paper topic: Emotion Recognition in Conversation (ERC).
+ Multi-Still: A lightweight multi-modal cross attention knowledge distillation method for real-time emotion recognition.

      * What is Emotion Recognition in Conversation?
      It is a research field focused on recognizing or predicting the emotions of participants during a conversation involving two or more individuals.
      

> 😊 Data Utilization: Research using the ETRI Korean Emotion Dataset

- 📁  [KEMDy19 (Situation drama dataset for voice actors)](https://nanum.etri.re.kr/share/kjnoh/KEMDy19?lang=ko_KR)
- 📁  [KEMDy20 (Spontaneous speech dataset for the general public)](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)


> 😊 Environment
```
python version : python3.9
OS type : WSL
requires packages: {
      'numpy==1.22.3',
      'pandas==1.4.2',
      'torch==1.11.0+cu113',
      'torchaudio==0.11.0+cu113',
      'scikit-learn',
      'transformers==4.18.0',
      'tokenizers==0.12.1',
      'soundfile==0.10.3.post1'
}
```

> 😊 Docker container run
```bash
docker container run -d -it --name multi_still --gpus all python:3.9
```

> 😊 Environment Setting
```bash
git clone https://github.com/SeolRoh/Multi-Still_ETRI.git
cd Multi-Still_ETRI
bash setup.sh
```
> 😊 Preprocessing
```bash
# Data preprocessing
bash data_preprocessing.sh
```

+ Distribution comparison before and after mitigating data imbalance in 7 emotion labels

![](https://github.com/SeolRoh/Multi-Still_ETRI/blob/main/images/datapreprocessing.png)


> 😊 Train
```bash
# Train the multi-modal teacher model
python train_crossattention.py --model_name multimodal_teacher

# Use the teacher model to add distilled data (Softmax) to the dataset
# Use the --teacher_name option to input the name_epoch of the MultiModal teacher model.
# Use the --data_path option to input the path of the existing dataset to add softmax data (default, "data/train_preprocessed_data.json")
python Distill_knowledge.py --teacher_name multimodal_teacher_epoch4 

# Modify miniconfig.py to change hyperparameters, including Epoch
# Train the multi-modal student model with knowledge distillation
python KD_train_crossattention.py --model_name multimodal_student
# Train the text-only student model with knowledge distillation
python KD_train_crossattention.py --model_name text_student --text_only True 
# Train the audio-only student model with knowledge distillation
python KD_train_crossattention.py --model_name audio_student --audio_only True
```

> 😊 Test
```bash
# pt files are generated every 5th Epoch during training (e.g., 5, 10, 11....)
# Copy to the test_all folder to test multiple files
mkdir ckpt/test_all
cp ckpt/* ckpt/test_all/
python test.py --all

# The result of test.py can be found in "result_all_model2.csv"
```


> 😁 Directory
- To implement the code, files provided by ETRI (KEMDy19 & KEMDy20) and AI Hub emotion data files must be in the correct locations.
```
+--Multi-Still_ETRI
      +--KEMDy19
            +--annotation
            +--EDA
            +--TEMP
            +--wav

      +--KEMDy20
            +--annotation
            +--wav
            +--TEMP
            +--IBI
            +--EDA 

      +--models
            +--module_for_clossattention
                  +--MultiheadAttention.py            # Multi-head attention
                  +--PositionalEmbedding.py           # Positional embedding
                  +--Transformer.py                   # Transformer
            +--multimodal.py                          # Multi-modal encoding
            +--multimodal_cross_attention.py          # Multi-modal encoding and cross attention including audio and text

      +--data (generated after running data_preprocessing.sh)
            +--total_data.json                        # Preprocessed file of all datasets
            +--preprocessed_data.json                 # File separating training and testing data, removing data without audio files
            +--test_preprocessed_data.json            # Extracted test data from preprocessed_data.json
            +--train_preprocessed_data.json           # Extracted training data from preprocessed_data.json

      +--ckpt (generated after running train_crossattention.py, KD_train_crossattention.py)
            +--test_all                               # Folder for copying multiple models for testing
            +--*.pt                                   # Model files saved every 5th Epoch during training

      +--TOTAL (generated after running Data_Preprocessing.sh)    # Preprocessed and trained data after copying all data to TOTAL folder
            +--hidden_states                          # Stored encoded results from pre-trained Wav2Vec2 model for faster training and inference.

      +--setup.sh                         # Script to update, upgrade, and install required libraries for model creation

      +--data_preprocessing.sh            # Script for data preprocessing and separating training and testing datasets

      +--config.py                        # Defines hyperparameters needed for training the teacher model

      +--Data_Balancing.py                # Script for data preprocessing and separating training and testing datasets

      +--Distill_knowledge.py             # Adds distilled knowledge (Softmax data) to the dataset using the trained teacher model

      +--KD_train_crossattention.py       # Trains the student model using distilled knowledge

      +--KEMDy_preprocessing.py           # Moves all data to the TOTAL folder and processes it into a dataset

      +--merdataset.py                    # DataLoader file that reads and provides data for model training

      +--mini_config.py                   # Defines hyperparameters needed for training the student model

      +--test.py                          # Tests models saved in the ckpt folder
      
      +--train_crossattention.py          # Script for training the cross attention-based teacher model

      +--utils.py                         # Includes functions to view model information such as logger and get_params
```

> 😆 Base Model

Encoder | Architecture | Pretrained Weights
:------------: | :-------------: | :-------------:
Audio Encoder | pretrained Wav2Vec 2.0 | kresnik/wav2vec2-large-xlsr-korean
Text Encoder | pretrained Electra | monologg/koelectra-base 

> 😃 Arguments
- train_crossattention.py

Arguments | Description
:------------: | :-------------:
--epochs | Number of model training iterations
--batch | Data batch size
--shuffle | Whether to shuffle training data
--lr | Learning rate value for training
--cuda | GPU to use (default="cuda:0")
--save | Whether to save the model
--model_name | Name to save the model
--text_only | Train using only text data and encoder
--audio_only | Train using only audio data and encoder

- Distill_knowledge.py

Arguments | Description
:------------: | :-------------:
--cuda | GPU to use (default="cuda:0")
--teacher_name | Name of the saved model to distill knowledge from
--data_path | Path to the dataset to distill knowledge

- KD_train_crossattention.py

Arguments | Description
:------------: | :-------------:
--epochs | Number of model training iterations
--batch | Data batch size
--shuffle | Whether to shuffle training data
--lr | Learning rate value for training
--cuda | GPU to use (default="cuda:0")
--save | Whether to save the model
--model_name | Name to save the model
--text_only | Train using only text data and encoder
--audio_only | Train using only audio data and encoder

- test.py

Arguments | Description
:------------: | :-------------:
--batch | Data batch size
--cuda | GPU to use (default="cuda:0")
--model_name | Name of the model to test (e.g., ckpt/[model_name].pt)
--all | Test all models in the "ckpt/test_all" path


> 😀 Model Architecture
- `Multi-Still` uses knowledge distillation to lighten the multi-modal structure for real-time emotion recognition.
- 👩‍🏫➡👨‍💻 Multi-Still Architecture
![](https://github.com/SeolRoh/Multi-Still_ETRI/blob/main/images/structure.png)
- 👩‍🏫 Teacher Model
![](https://github.com/SeolRoh/Multi-St