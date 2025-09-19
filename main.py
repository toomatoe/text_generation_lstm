import tensorflow as tf
import pandas as pd
import numpy as np
import random
import sys

df = pd.read_csv('train.csv')
text = " ".join(df['text'].dropna().astype(str)).lower()

print(f"Total characters in text: `{len(text)}`")