import pandas as pd
import numpy as np

# Load original IRIS data
df = pd.read_csv("data/v1data.csv")

# Task 1: Add a random sensitive attribute "location" (0 or 1)
np.random.seed(42)
df['location'] = np.random.randint(0, 2, size=len(df))

# Save this new version — this is what we'll use for the rest of the assignment
df.to_csv("data/iris_with_location.csv", index=False)

print(df.head(10))
print("\nLocation value counts:")
print(df['location'].value_counts())