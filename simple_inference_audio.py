import torch

from models.multimodal import TextEncoder, SpeechEncoder
from merdataset import *
from config import *

text_config['cuda'] = 'cuda:0'
test_config['cuda'] = 'cuda:0'


def test(model, batch_x):
    model.eval()

    with torch.no_grad():
        print(batch_x)
        outputs = model(batch_x)  # ,do_clf=args.do_clf)
        print("outputs:",outputs)


import os
import torch
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2Config

def process_wav_to_pt(wav_file_path, output_dir, device='cuda'):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Configuration for Wav2Vec 2.0
    config = Wav2Vec2Config(
        num_hidden_layers=6,
        hidden_size=1024,
        output_hidden_size=1024,
        num_attention_heads=16
    )
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model(config).to(device)

    # Load the WAV file
    wav, _ = sf.read(wav_file_path)

    # Processing
    with torch.no_grad():
        input_values = processor(wav, return_tensors='pt', padding=True, sampling_rate=16000).input_values
        input_values = input_values.to(device)
        outputs = model(input_values)
        hidden_states = outputs.last_hidden_state

    # Save the output tensor
    output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(wav_file_path))[0] + '.pt')
    torch.save(hidden_states, output_file_path)
    print(f'Saved processed tensor to {output_file_path}')

    # Clean up
    del model
    torch.cuda.empty_cache()
    return output_file_path




if __name__ == '__main__':
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    model = torch.load('./ckpt/audio_student_epoch0.pt')
    #model = torch.load('./ckpt/text_student_epoch94.pt')
    """
    training data = {'file_name': 'Sess14_script03_M009', 'wav': 'wav_Sess14_script03_M009.wav', 'utterance': 'b/ 자기 지금 무슨 소리야. 농담이지?\n', 'Emotion': ['sad', 'neutral', 'neutral', 'sad', 'sad', 'sad', 'neutral', 'sad', 'sad', 'sad', 'fee', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise', 'surprise'], 'label': [0.15, 0.0, 0.45, 0.0, 0.35, 0.0, 0.05], 'dialogue': 'b/ 자기 지금 무슨 소리야. 농담이지?\n'}
    """
    """
    test_data data = {'wav': 'wav_Sess14_script03_M009.wav', 'utterance': 'b/ 자기 지금 무슨 소리야. 농담이지?\n', 'dialogue': 'b/ 자기 지금 무슨 소리야. 농담이지?\n'}
    """
    #test(model, [{'wav': 'wav_Sess14_script03_M009.wav', 'utterance': 'b/ 자기 지금 무슨 소리야. 농담이지?\n', 'dialogue': 'b/ 자기 지금 무슨 소리야. 농담이지?\n'}])

    # Example usage
    ptpath = process_wav_to_pt('./KEMDy20/wav/Session14/Sess14_script01_User027F_001.wav', './tmp')
    test(model, [{'pt': ptpath}])
    #python KD_train_crossattention.py --model_name audio_student --audio_only True --epoch 1
    #test(model, [{'utterance': 'b/ 자기 지금 무슨 소리야. 농담이지?\n', 'dialogue': 'b/ 자기 지금 무슨 소리야. 농담이지?\n'}])

