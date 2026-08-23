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
    token = result.stdout.strip()
    print(f"Token length: {len(token)}")
    return token

def predict(endpoint_id, input_text):
    token = get_token()
    url = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_NUMBER}/locations/{LOCATION}/endpoints/{endpoint_id}:generateContent"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": input_text}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10}
    }
    response = requests.post(url, headers=headers, json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text[:200]}")
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

# Test with just first sample
print("Testing V1 with first sample...")
test_row = test_df.iloc[0]
input_text = f"sepal_length: {test_row['sepal_length']}, sepal_width: {test_row['sepal_width']}, petal_length: {test_row['petal_length']}, petal_width: {test_row['petal_width']}. Reply with only one word: setosa, versicolor, or virginica."
result = predict(V1_ENDPOINT, input_text)
print(f"Result: {result}")
print(f"Actual: {test_row['species']}")
