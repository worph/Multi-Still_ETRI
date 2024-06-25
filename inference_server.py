import torch
from flask import Flask, request, jsonify
from transformers import ElectraTokenizer

from flask_cors import CORS  # Import CORS

# Load the trained model
model_path = './ckpt/text_student_epoch67.pt'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the entire model
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Initialize the tokenizer
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

# Define the emotion labels
emotion_labels = ['neutral', 'happy', 'surprise', 'angry', 'sad', 'disgust', 'fear']


# Define a function to preprocess input text data
def preprocess_text(text):
    # Tokenize the input text
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    return tokens


# Define a function to perform inference
def predict_emotions(text):
    # Create a batch dictionary
    batch = [{'dialogue': text}]

    with torch.no_grad():
        output = model(batch)

    # Get the most likely emotion
    emotion_probabilities = output[0].tolist()
    # print(emotion_probabilities)
    # Create a map of emotion labels to probabilities
    emotion_probabilities = {emotion: round(probability, 3) for emotion, probability in
                             zip(emotion_labels, emotion_probabilities)}

    # filter out the emotion that are above 0.1
    # alpha = 0.001
    # emotion_probabilities = {emotion: probability for emotion, probability in emotion_probabilities.items() if probability > alpha}

    return emotion_probabilities


# Create a Flask application
app = Flask(__name__)
CORS(app)  # Enable CORS on the app, allowing all domains


# Define a route for the inference
@app.route('/predict', methods=['GET'])
def predict():
    # Get the text from the request
    text = request.args.get('text')
    # Predict the emotion
    emotions = predict_emotions(text)
    print("Predicted " + text + "emotions :", emotions)

    # Return the result as JSON
    return jsonify(emotions)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
