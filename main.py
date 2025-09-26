import tensorflow as tf
import pandas as pd
import numpy as np
import random
import sys

df = pd.read_csv('train.csv')
text = " ".join(df['text'].dropna().astype(str)).lower()

print(f"Total characters in text: `{len(text)}`")
vocab = sorted(set(text))
print(f"Vocabulary size: `{len(vocab)}`")

char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]
dataset = sequences.map(split_input_target)

BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
