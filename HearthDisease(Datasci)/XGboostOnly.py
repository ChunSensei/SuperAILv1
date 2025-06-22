import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, fbeta_score
from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("C:/Users/User/Desktop/1Year/Term2/SuperEngi/HearthDisease/hearth-disease-recognition/train.csv")

columns_to_drop = ["ID", "Cholesterol Checked", "Health Care Coverage", 
                   "Doctor Visit Cost Barrier", "Income Level", "Education Level"]
df = df.drop(columns=columns_to_drop)

df = df.dropna()
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

y = df["History of HeartDisease or Attack"]  
X = df.drop(columns=["History of HeartDisease or Attack"]) 

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

model = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
f2_score = fbeta_score(y_test, y_pred, beta=2)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", report)

test_df = pd.read_csv("C:/Users/User/Desktop/1Year/Term2/SuperEngi/HearthDisease/hearth-disease-recognition/test.csv")
test_df = test_df.drop(columns=columns_to_drop)
test_df = test_df.dropna()

test_encoded = test_df.copy()
for col in test_encoded.select_dtypes(include=['object']).columns:
    if col in label_encoders:
        test_encoded[col] = label_encoders[col].transform(test_encoded[col])

test_predictions = model.predict(test_encoded)

submission_labels = ["No" if pred == 0 else "Yes" for pred in test_predictions]

submission = pd.read_csv("C:/Users/User/Desktop/1Year/Term2/SuperEngi/HearthDisease/hearth-disease-recognition/sample_submission.csv")
submission["History of HeartDisease or Attack"] = submission_labels
submission.to_csv("C:/Users/User/Desktop/1Year/Term2/SuperEngi/HearthDisease/hearth-disease-recognition/submission.csv", index=False)

print("Submission file saved as submission.csv")
