import pandas as pd

df = pd.read_csv('data/isl_landmarks.csv')
df = df[df['target'] < 26]
df.to_csv('data/isl_landmarks.csv', index=False)
print(df['target'].value_counts().sort_index())