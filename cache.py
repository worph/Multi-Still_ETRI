from transformers import ElectraTokenizer

# cache the model
ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")