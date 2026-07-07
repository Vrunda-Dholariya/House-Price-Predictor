from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load a real housing dataset that comes built into scikit-learn
data=fetch_california_housing(as_frame=True)
df=data.frame

print("Shape (rows, columns):", df.shape)
print()
print("First 5 rows:")
print(df.head())
print("Column names:",df.columns.tolist())