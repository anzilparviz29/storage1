import onnxruntime as ort
from PIL import Image
import numpy as np
from my_tokenizer import GPT2Tokenizer

# Load tokenizer
tokenizer = GPT2Tokenizer("gpt2_tokenizer/vocab.json", "gpt2_tokenizer/merges.txt")

# Load ONNX models
encoder_sess = ort.InferenceSession("vit_encoder.onnx")
decoder_sess = ort.InferenceSession("gpt2_decoder.onnx")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_np = (np.array(image).astype(np.float32) / 255.0 - 0.5) / 0.5
    image_np = image_np.transpose(2, 0, 1)  # HWC to CHW
    return np.expand_dims(image_np, axis=0)

pixel_values = preprocess_image("myimage1.jpg")

encoder_hidden_states = encoder_sess.run(None, {"pixel_values": pixel_values})[0]

start_token_id = 50256  # Use GPT-2's  token consistently

input_ids = [start_token_id]
attention_mask = [1]

for _ in range(20):
    inputs = {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "encoder_hidden_states": encoder_hidden_states,
        "attention_mask": np.array([attention_mask], dtype=np.int64),
    }
    logits = decoder_sess.run(None, inputs)[0]
    next_token = np.argmax(logits[0, -1, :])
    input_ids.append(next_token)
    attention_mask.append(1)
    if next_token == start_token_id:
        break

caption = tokenizer.decode(input_ids[1:])  # remove the start token for clean output
print("Caption:", caption)
