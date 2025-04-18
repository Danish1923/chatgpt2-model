# 🧠 GPT2-Style Transformer in TensorFlow

This repository contains a clean and modular implementation of a GPT-2 style Transformer model built using TensorFlow and Keras. It includes core components such as Multi-Head Self Attention, Feed Forward Networks, Transformer Blocks, and causal masking, all stitched into a functional GPT2-like model.

---

## 🚀 Features

- ✅ Multi-Head Self Attention
- ✅ Feed Forward Network
- ✅ Positional Embeddings
- ✅ Layer Normalization & Dropout
- ✅ Causal Masking for autoregressive tasks
- ✅ Easily customizable for number of layers, heads, embedding size

---

## 🛠️ Requirements

Make sure to install the following before running the model:

```bash
pip install tensorflow


📦 Model Architecture Overview

Embedding Layer – Token & Positional Embeddings

Transformer Blocks – Stack of Multi-head Attention + Feed Forward

Causal Masking – Ensures autoregressive flow

Output Layer – Dense layer mapping to vocabulary


🧰 How to Build & Run
Follow these simple steps:

1️⃣ Clone the Repository

git clone https://github.com/your-username/gpt2-tf-from-scratch.git
cd gpt2-tf-from-scratch

2️⃣ Install Dependencies

pip install tensorflow

3️⃣ Run the Model 🏃‍♂️
Inside your Python environment or script:

import tensorflow as tf
from model import GPT2  # Assuming you save the code in model.py


VOCAB_SIZE = 50257
MAX_LENGTH = 1024


model = GPT2(vocab_size=VOCAB_SIZE, max_length=MAX_LENGTH)


dummy_input = tf.random.uniform((2, 50), maxval=VOCAB_SIZE, dtype=tf.int32)
output = model(dummy_input)

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)

🧪 Testing
You can test the model using a dummy input batch:

dummy_input = tf.random.uniform((2, 50), maxval=50257, dtype=tf.int32)
output_logits = model(dummy_input)
print("Output shape:", output_logits.shape)  # (2, 50, 50257)

📊 Model Summary

model.summary()

🔧 Customization
You can easily change hyperparameters when instantiating the model:

GPT2(
  vocab_size=50257,
  max_length=1024,
  embed_dim=512,
  num_heads=8,
  dff=2048,
  num_layers=6
)


📚 References

GPT-2 Paper (OpenAI)
TensorFlow Keras Documentation

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

❤️ Contributions
Feel free to open issues, suggest features, or make pull requests! PRs are always welcome.

✨ Happy Coding!


---

Let me know if you want a version with badges, examples, or formatted as a GitHub template repo!
