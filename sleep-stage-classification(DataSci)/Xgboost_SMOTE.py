import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.combine import SMOTETomek
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft

# Paths
train_folder = r"C:\Users\User\Desktop\1Year\Term2\SuperEngi\sleepppppAI\train\train"
test_folder = r"C:\Users\User\Desktop\1Year\Term2\SuperEngi\sleepppppAI\test_segment\test_segment\test001"
sample_submission_path = r"C:\Users\User\Desktop\1Year\Term2\SuperEngi\sleepppppAI\sample_submission.csv"
output_submission_path = r"C:\Users\User\Desktop\1Year\Term2\SuperEngi\sleepppppAI\submission.csv"

# Load training data
def load_train_data(train_folder):
    all_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.csv')]
    df_list = []
    
    for i, f in enumerate(tqdm(all_files, desc="Loading Training Data")):
        df = pd.read_csv(f)
        df['file_id'] = i
        df_list.append(df)
    
    train_df = pd.concat(df_list, ignore_index=True)
    return train_df

train_df = load_train_data(train_folder)

# Features & Target
features = ['BVP', 'ACC_X', 'ACC_Y', 'ACC_Z', 'TEMP', 'EDA', 'HR', 'IBI']
X = train_df[features]
y = train_df['Sleep_Stage']

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature Engineering
SEGMENT_SIZE = 480

def extract_features(df, segment_size=SEGMENT_SIZE):
    feature_list = []
    label_list = []
    
    for file_id in df['file_id'].unique():
        file_df = df[df['file_id'] == file_id].reset_index(drop=True)
        num_segments = len(file_df) // segment_size
        
        for i in range(num_segments):
            segment = file_df.iloc[i * segment_size : (i + 1) * segment_size]
            feature_vector = {f"{col}_mean": segment[col].mean() for col in features}
            feature_vector.update({f"{col}_std": segment[col].std() for col in features})
            feature_vector.update({f"{col}_min": segment[col].min() for col in features})
            feature_vector.update({f"{col}_max": segment[col].max() for col in features})
            feature_vector.update({f"{col}_median": segment[col].median() for col in features})
            
            # Calculate skewness and kurtosis only if the data is not constant
            for col in features:
                if segment[col].std() > 1e-10:  # Check if the standard deviation is not too small
                    feature_vector[f"{col}_skew"] = skew(segment[col])
                    feature_vector[f"{col}_kurtosis"] = kurtosis(segment[col])
                else:
                    feature_vector[f"{col}_skew"] = 0.0  # Default value for constant data
                    feature_vector[f"{col}_kurtosis"] = 0.0  # Default value for constant data
            
            # Frequency domain features (FFT)
            for col in features:
                fft_values = np.abs(fft(segment[col].values))
                feature_vector[f"{col}_fft_mean"] = np.mean(fft_values)
                feature_vector[f"{col}_fft_std"] = np.std(fft_values)
            
            feature_list.append(feature_vector)
            label_list.append(segment['Sleep_Stage'].iloc[0])
    
    return pd.DataFrame(feature_list), np.array(label_list)

X_features, y_labels = extract_features(train_df)
y_labels_encoded = label_encoder.transform(y_labels)

# Balance Data (ใช้ SMOTE + Tomek Links)
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_features, y_labels_encoded)

# Standardize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)

# Train XGBoost Model (ปรับ Hyperparameters ใหม่)
xgb_model = XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=6, 
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=3, random_state=42)
xgb_model.fit(X_train, y_train)

# Validate Model
y_pred = xgb_model.predict(X_val)

# Evaluate Model
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred, average='weighted')
report = classification_report(y_val, y_pred, target_names=label_encoder.classes_)

print(f"Validation Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Classification Report:\n", report)

# Load Sample Submission
sample_submission = pd.read_csv(sample_submission_path)

def process_test_file(sample_id):
    file_path = os.path.join(test_folder, f"{sample_id}.csv")
    if os.path.exists(file_path):
        test_data = pd.read_csv(file_path)
        
        test_features = {f"{col}_mean": test_data[col].mean() for col in features}
        test_features.update({f"{col}_std": test_data[col].std() for col in features})
        test_features.update({f"{col}_min": test_data[col].min() for col in features})
        test_features.update({f"{col}_max": test_data[col].max() for col in features})
        test_features.update({f"{col}_median": test_data[col].median() for col in features})
        
        # Calculate skewness and kurtosis only if the data is not constant
        for col in features:
            if test_data[col].std() > 1e-10:  # Check if the standard deviation is not too small
                test_features[f"{col}_skew"] = skew(test_data[col])
                test_features[f"{col}_kurtosis"] = kurtosis(test_data[col])
            else:
                test_features[f"{col}_skew"] = 0.0  # Default value for constant data
                test_features[f"{col}_kurtosis"] = 0.0  # Default value for constant data
        
        # Frequency domain features (FFT)
        for col in features:
            fft_values = np.abs(fft(test_data[col].values))
            test_features[f"{col}_fft_mean"] = np.mean(fft_values)
            test_features[f"{col}_fft_std"] = np.std(fft_values)
        
        test_features_df = pd.DataFrame([test_features])
        X_test = scaler.transform(test_features_df)
        
        # Use predict_proba to get probabilities
        y_pred_proba = xgb_model.predict_proba(X_test)
        predicted_label = label_encoder.inverse_transform(np.argmax(y_pred_proba, axis=1))[0]
        return pd.DataFrame([{'id': sample_id, 'labels': predicted_label}])
    return None

# Predict on Test Data using Multi-threading
submission_results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(process_test_file, sample_submission['id'].unique()), total=len(sample_submission['id'].unique()), desc="Processing Test Data"))

submission_results = pd.concat([df for df in results if df is not None], ignore_index=True)

# Save Submission
submission_results.to_csv(output_submission_path, index=False, header=True)
print(f"Submission file saved as {output_submission_path}")