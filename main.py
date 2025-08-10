import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve,roc_auc_score

# 1. Load datasets
train_data = pd.read_csv('Dataset.txt', sep='\t')
test_data = pd.read_csv('Dataset_test.txt', sep='\t')

# 2. Feature Engineering
def feature_engineer(data, is_training=True):
    data_processed = data.copy()
    date_features = ['F15','F16']
    for date_col in date_features:
        data_processed[date_col] = pd.to_datetime(data_processed[date_col], errors = 'coerce')
        data_processed[f'{date_col}_year'] = data_processed[date_col].dt.year
        data_processed[f'{date_col}_month'] = data_processed[date_col].dt.year
        data_processed[f'{date_col}_day'] = data_processed[date_col].dt.year
        # for my reference
        # Days from epoch is used to capture the time difference in a numeric format
        # This helps in capturing trends over time. It is standard to use 1970-01-01 as the epoch start date.
        data_processed[f'{date_col}_days_since_epoch'] = (data_processed[date_col] - pd.Timestamp('1970-01-01')).dt.days 
        
    if len(date_features) >= 2:
        data_processed['date_diff_days'] = (data_processed[date_features[1]] - data_processed[date_features[0]]).dt.days

    date_derived_cols = []
    for date_col in date_features:
        date_derived_cols.extend([f'{date_col}_year', f'{date_col}_month', f'{date_col}_day', f'{date_col}_days_since_epoch'])
    date_derived_cols.append('date_diff_days')
    for col in date_derived_cols:
        if col in data_processed.columns:
            data_processed[col] = data_processed[col].fillna(data_processed[col].median())
    columns_to_drop = date_features + (['Index'] if is_training else ['Index'])
    data_processed = data_processed.drop(columns=columns_to_drop)
    
    return data_processed

train_processed = feature_engineer(train_data, is_training=True)
test_processed = feature_engineer(test_data, is_training=False)

# 3. Model Training
feature_columns = [col for col in train_processed.columns if col != 'C']
X = train_processed[feature_columns]
y = train_processed['C']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# 4. Predicting and evaluating
val_predictions = rf.predict(X_val)
val_probabilities = rf.predict_proba(X_val)[:, 1]

# Metrics
accuracy = accuracy_score(y_val, val_predictions)
auc = roc_auc_score(y_val, val_probabilities)
print(f"Validation Accuracy: {accuracy:.4f}")


# Training predictions
train_predictions = rf.predict(X)
test_predictions = rf.predict(test_processed[feature_columns])

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, val_probabilities)
axes[0, 0].plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_val, val_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title('Confusion Matrix')

# Target Distribution
y.value_counts().plot(kind='bar', ax=axes[1, 0])
axes[1, 1].set_xlabel('Class')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Target Variable Distribution')
axes[1, 1].tick_params(axis='x', rotation=0)
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()

# Saving predictions
train_pred_df = pd.DataFrame({
    'Index': train_data['Index'].values,
    'Class': train_predictions
})
train_pred_df.to_csv('training_predictions.txt', sep='\t', index=False)
test_pred_df = pd.DataFrame({
    'Index': test_data['Index'].values,
    'Class': test_predictions
})
test_pred_df.to_csv('test_predictions.txt', sep='\t', index=False)
