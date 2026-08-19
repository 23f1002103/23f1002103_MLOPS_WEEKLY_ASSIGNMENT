import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the model and test data saved from Task 2
clf = joblib.load("iris_rf_model.bin")
X_test = pd.read_csv("X_test.csv")

# Class names in the order the model learned them
class_names = clf.classes_
print("Class order:", class_names)

# Create SHAP explainer for the trained RandomForest
explainer = shap.Explainer(clf, X_test)
shap_values = explainer(X_test)

print("SHAP values shape:", shap_values.values.shape)  # (samples, features, classes)

# Generate and save a summary plot for EACH class
for i, name in enumerate(class_names):
    plt.figure()
    shap.summary_plot(shap_values[:, :, i], X_test, show=False)
    plt.title(f"SHAP Summary — {name}")
    plt.tight_layout()
    plt.savefig(f"shap_summary_{name}.png")
    plt.close()
    print(f"Saved shap_summary_{name}.png")