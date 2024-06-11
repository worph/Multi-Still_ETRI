import torch
import pandas as pd
from transformers import ElectraTokenizer
from mini_config import audio_config, text_config, cross_attention_config

# Load the trained model
model_path = './ckpt/text_student_epoch4.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the entire model
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Initialize the tokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# Define the emotion labels
emotion_labels = ["angry", "sad", "happy", "disgust", "fear", "surprise", "neutral"]


# Define a function to preprocess input text data
def preprocess_text(text):
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    return tokens


# Define a function to perform inference
def predict_emotion(text):
    # Create a batch dictionary
    batch = [{'dialogue': text}]

    with torch.no_grad():
        output = model(batch)

    # Get the most likely emotion
    emotion_probabilities = output[0]
    print(emotion_probabilities)
    most_likely_emotion_index = torch.argmax(emotion_probabilities).item()
    most_likely_emotion = emotion_labels[most_likely_emotion_index]

    return most_likely_emotion


print("Predicted emotion:"+predict_emotion('켁켁 어? 갑자기 뭔 꿈?\n'))  # Expected: happy
print(f"Predicted emotion: {predict_emotion('어젯밤에 무서운 꿈을 꾸어서 잠을 설쳤어요.')}")  # Expected: fear
print(f"Predicted emotion: {predict_emotion('친구와 크게 싸워서 정말 화가 났어요.')}")  # Expected: angry
print(f"Predicted emotion: {predict_emotion('어제 본 영화가 너무 슬펐어요.')}")  # Expected: sad
print(f"Predicted emotion: {predict_emotion('음식이 너무 맛있어서 깜짝 놀랐어요.')}")  # Expected: surprise
print(f"Predicted emotion: {predict_emotion('지금 기분이 그냥 그래요.')}")  # Expected: neutral
print(f"Predicted emotion: {predict_emotion('길을 가다가 넘어져서 정말 창피했어요.')}")  # Expected: disgust
print(f"Predicted emotion: {predict_emotion('오늘 회사에서 좋은 소식을 들어서 기분이 좋아요.')}")  # Expected: happy
print(f"Predicted emotion: {predict_emotion('밤에 혼자 있어서 조금 무서웠어요.')}")  # Expected: fear
print(f"Predicted emotion: {predict_emotion('시험 결과가 나왔는데 만족스럽지 않아서 화가 났어요.')}")  # Expected: angry
print(f"Predicted emotion: {predict_emotion('오랜만에 친구를 만나서 기뻤어요.')}")  # Expected: happy
print(f"Predicted emotion: {predict_emotion('길에서 갑자기 차가 튀어나와서 깜짝 놀랐어요.')}")  # Expected: surprise
print(f"Predicted emotion: {predict_emotion('그냥 평범한 하루였어요.')}")  # Expected: neutral
print(f"Predicted emotion: {predict_emotion('이 음식은 정말 못 먹겠어요.')}")  # Expected: disgust
print(f"Predicted emotion: {predict_emotion('오늘 기분이 우울해요.')}")  # Expected: sad
print(f"Predicted emotion: {predict_emotion('친구가 놀러와서 정말 기뻤어요.')}")  # Expected: happy
print(f"Predicted emotion: {predict_emotion('늦은 밤에 혼자 있어서 조금 무서워요.')}")  # Expected: fear
print(f"Predicted emotion: {predict_emotion('회의 중에 실수를 해서 정말 화가 났어요.')}")  # Expected: angry
print(f"Predicted emotion: {predict_emotion('좋아하는 밴드의 공연을 봐서 정말 기뻤어요.')}")  # Expected: happy
print(f"Predicted emotion: {predict_emotion('아무 일도 없어서 그냥 평범한 하루였어요.')}")  # Expected: neutral
