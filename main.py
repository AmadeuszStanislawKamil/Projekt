import pandas as pd
import numpy as np

df = pd.read_csv('PlantData/GenerationData/Plant_1_Generation_Data.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]

