import os

import torch
from transformers import ElectraTokenizer

# Load the trained model
model_path = './ckpt/text_student_epoch67.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('cuda available:',torch.cuda.is_available())
print('env:',os.environ)

# Load the entire model
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Initialize the tokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# Define the emotion labels
emotion_labels = ['neutral', 'happy', 'surprise', 'angry', 'sad', 'disgust', 'fear']

# Define a function to perform inference
def predict_emotions(text):
    # Create a batch dictionary
    """
    {
    'file_name': 'Sess19_script02_User037M_015',
    'wav': 'wav_Sess19_script02_User037M_015.wav',
    'utterance': 'c/ 거기에 대해서 딱 5 대 5로 뽑아야 된다 막 이러는 거 진짜 아닌 거지 그치.\n',
    'Emotion': ['disgust', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'neutral', 'angry', 'disgust', 'angry'],
    'label': [0.6, 0.0, 0.0, 0.2, 0.0, 0.2, 0.0],
    'dialogue': 'c/ 거기에 대해서 딱 5   대 5로 뽑아야 된다 막 이러는 거는 진짜 아닌 거지 그치.\n'
    }
    """
    batch = [{
        'dialogue': "c/ "+text+"\n",
        'utterance': "c/ "+text+"\n",
    }]

    with torch.no_grad():
        outputs = model(batch)

    # Get the most likely emotion
    emotion_probabilities = outputs[0].tolist()
    #emotion_probabilities = outputs.max(dim=1)[1].tolist()
    #Create a map of emotion labels to probabilities
    emotion_probabilities = {emotion: probability for emotion, probability in zip(emotion_labels, emotion_probabilities)}
    #filter out the emotion that are above 0.1
    alpha = 0.01
    emotion_probabilities = {emotion: probability for emotion, probability in emotion_probabilities.items() if probability > alpha}

    return emotion_probabilities


print("Predicted emotions:", predict_emotions("한번 드셔보세요. 맛있을 거에요.\n"))  # Expected: angry
print("Predicted emotions:", predict_emotions("(이래+자꾸) 이러면 나도 더 이상 안 참아. \n"))  # Expected: fear
print("Predicted emotions:", predict_emotions("아 그런 남자들 딱 질색이야.\n"))  # Expected: angry
print("Predicted emotions:", predict_emotions("집안 문제는 뭐 어떻게 보면 고칠 수 없는 거니까. \n"))  # Expected: sad
print("Predicted emotions:", predict_emotions("허어 아 진, 진짜로 나 이거 처음 듣는 건데?\n"))  # Expected: surprise
print("Predicted emotions:", predict_emotions("뭐 남자라서 차별받았다던가.\n"))  # Expected: neutral
print("Predicted emotions:", predict_emotions("l/ 그 아니 신랑 선배 중에도 진짜 꼰대가 있었어.\n"))  # Expected: disgust
print("Predicted emotions:", predict_emotions('오늘 회사에서 좋은 소식을 들어서 기분이 좋아요.'))  # Expected: happy
print("Predicted emotions:", predict_emotions('밤에 혼자 있어서 조금 무서웠어요.'))  # Expected: fear
print("Predicted emotions:", predict_emotions('시험 결과가 나왔는데 만족스럽지 않아서 화가 났어요.'))  # Expected: angry
print("Predicted emotions:", predict_emotions('오랜만에 친구를 만나서 기뻤어요.'))  # Expected: happy
print("Predicted emotions:", predict_emotions('길에서 갑자기 차가 튀어나와서 깜짝 놀랐어요.'))  # Expected: surprise
print("Predicted emotions:", predict_emotions('그냥 평범한 하루였어요.'))  # Expected: neutral
print("Predicted emotions:", predict_emotions('이 음식은 정말 못 먹겠어요.'))  # Expected: disgust
print("Predicted emotions:", predict_emotions('오늘 기분이 우울해요.'))  # Expected: sad
print("Predicted emotions:", predict_emotions('친구가 놀러와서 정말 기뻤어요.'))  # Expected: happy
print("Predicted emotions:", predict_emotions('늦은 밤에 혼자 있어서 조금 무서워요.'))  # Expected: fear
print("Predicted emotions:", predict_emotions('회의 중에 실수를 해서 정말 화가 났어요.'))  # Expected: angry
print("Predicted emotions:", predict_emotions('좋아하는 밴드의 공연을 봐서 정말 기뻤어요.'))  # Expected: happy
print("Predicted emotions:", predict_emotions('아무 일도 없어서 그냥 평범한 하루였어요.'))  # Expected: neutral
