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