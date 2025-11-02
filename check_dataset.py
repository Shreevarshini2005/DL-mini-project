import pandas as pd
df = pd.read_csv("data/mbti_1.csv")
print(df.head())  # check the first few rows
print(df['type'].value_counts())  # see number of samples per MBTI type
