import pandas as pd
import numpy as np

np.random.seed(42)

n = 200

data = {
    "Math": np.random.randint(50, 100, n),
    "Programming": np.random.randint(50, 100, n),
    "Communication": np.random.randint(50, 100, n),
    "Logic": np.random.randint(50, 100, n)
}

df = pd.DataFrame(data)

career = []

for _, row in df.iterrows():

    if row["Programming"] > 85 and row["Logic"] > 80:
        career.append("Software Engineer")

    elif row["Math"] > 85 and row["Logic"] > 85:
        career.append("Data Scientist")

    elif row["Communication"] > 85:
        career.append("Teacher")

    else:
        career.append("HR")

df["Career"] = career

df.to_csv("../DATA/student_data.csv", index=False)

print("Dataset created successfully with 200 students.")