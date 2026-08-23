import requests
import pandas as pd
from sklearn.metrics import accuracy_score
import sys
import subprocess

# Config
PROJECT_NUMBER = "927061930480"
LOCATION = "us-central1"
V1_ENDPOINT = "2976708379633778688"
V2_ENDPOINT = "6201707925296119808"
ACCURACY_THRESHOLD = 0.60

def get_token():
    result = subprocess.run(
        ['gcloud', 'auth', 'print-access-token'],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def predict(endpoint_id, input_text):
    token = get_token()
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}/locations/{LOCATION}/endpoints/{endpoint_id}:generateContent"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": input_text}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10}
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        text = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().lower()
        for species in ['setosa', 'versicolor', 'virginica']:
            if species in text:
                return species
        return "error"
    except:
        return "error"

# Load test data
test_df = pd.read_csv('iris_test.csv')
actual = list(test_df['species'])

# V1 predictions
print("Running V1 evaluation...")
v1_preds = []
for _, row in test_df.iterrows():
    input_text = f"sepal_length: {row['sepal_length']}, sepal_width: {row['sepal_width']}, petal_length: {row['petal_length']}, petal_width: {row['petal_width']}. Reply with only one word: setosa, versicolor, or virginica."
    v1_preds.append(predict(V1_ENDPOINT, input_text))

# V2 predictions
print("Running V2 evaluation...")
v2_preds = []
for _, row in test_df.iterrows():
    input_text = f"A flower specimen has a sepal length of {row['sepal_length']} cm, sepal width of {row['sepal_width']} cm, petal length of {row['petal_length']} cm, and petal width of {row['petal_width']} cm. Identify the iris species. Reply with only one word: setosa, versicolor, or virginica."
    v2_preds.append(predict(V2_ENDPOINT, input_text))

# Calculate accuracy
valid = ['setosa', 'versicolor', 'virginica']
v1_clean = [p if p in valid else 'unknown' for p in v1_preds]
v2_clean = [p if p in valid else 'unknown' for p in v2_preds]

v1_acc = accuracy_score(actual, v1_clean)
v2_acc = accuracy_score(actual, v2_clean)

print(f"\nV1 Accuracy: {v1_acc:.2%}")
print(f"V2 Accuracy: {v2_acc:.2%}")
print(f"Threshold: {ACCURACY_THRESHOLD:.2%}")

if v1_acc < ACCURACY_THRESHOLD:
    print(f"❌ V1 accuracy {v1_acc:.2%} below threshold!")
    sys.exit(1)
if v2_acc < ACCURACY_THRESHOLD:
    print(f"❌ V2 accuracy {v2_acc:.2%} below threshold!")
    sys.exit(1)

print("✅ Both models passed accuracy threshold!")
