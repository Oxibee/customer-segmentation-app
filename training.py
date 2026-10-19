import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 1. Data Ingestion & Feature Selection
df = pd.read_csv('customer_churn_dataset-training-master.csv').dropna()


segmentation_features = ['Tenure', 'Usage Frequency', 'Total Spend']
X = df[segmentation_features]


# Take a sample for efficient training
X_sample = X.sample(n=20000, random_state=42)


# 2. Explicit Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)


# 3. Train Final K-Means Model (Optimal K=3)
optimal_k = 3
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
final_kmeans.fit(X_scaled)


# 4. Package model, scaler, and feature list into an artifact dictionary
artifact = {
    'model': final_kmeans,
    'scaler': scaler,
    'features': segmentation_features,
}


joblib.dump(artifact, 'kmeans_model_artifact.pkl')
print(
    'K-Means clustering model and scaler successfully saved as'
    ' kmeans_model_artifact.pkl!'
)
