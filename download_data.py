import os
import pandas as pd
from urllib.request import urlretrieve

os.makedirs('src/data/datasets', exist_ok=True)

# UCI Heart Disease (Cleveland)
# We will download the processed cleveland data
heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
df_heart = pd.read_csv(heart_url, names=heart_cols, na_values='?')
# We will leave missing values as NaN since the app's preprocessing handles missing values!
# Actually, target is the last column
df_heart['target'] = (df_heart['target'] > 0).astype(int)
df_heart.to_csv('src/data/datasets/heart_disease.csv', index=False)
print("Saved Heart Disease dataset:", df_heart.shape)

# Parkinson's Dataset
parkinsons_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
df_park = pd.read_csv(parkinsons_url)
# Drop the 'name' column which is just a string ID
df_park = df_park.drop('name', axis=1)
# target is 'status' (0 or 1). We should probably move status to the end so our generic csv loader works naturally, or specify target_col.
# Let's move status to the end.
cols = list(df_park.columns)
cols.remove('status')
cols.append('status')
df_park = df_park[cols]
df_park.to_csv('src/data/datasets/parkinsons.csv', index=False)
print("Saved Parkinson's dataset:", df_park.shape)
