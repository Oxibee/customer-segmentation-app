import joblib
import numpy as np
import pandas as pd
import streamlit as st


# 1. Configure Page Layout
st.set_page_config(
    page_title='Customer Segmentation Portal', page_icon='🎯', layout='centered'
)

# 2. Cached Resource Loader
@st.cache_resource
def load_artifact():
  return joblib.load('kmeans_model_artifact.pkl')

# Load artifact dictionary
artifact = load_artifact()
model = artifact['model']
scaler = artifact['scaler']
features = artifact['features']

# Define friendly persona descriptions for each cluster
cluster_descriptions = {
    0: (
        '🟢 **Segment 0 (Standard Users):** Moderate tenure, regular usage,'
        ' and average spend.'
    ),
    1: (
        '⭐ **Segment 1 (VIP / High-Value Customers):** High tenure, high usage'
        ' frequency, and top-tier total spend.'
    ),
    2: (
        '⚠️ **Segment 2 (At-Risk / Low-Engagement):** Low tenure, minimal usage,'
        ' and low spend.'
    ),
}


# 3. User Interface Design
st.title('🎯 Customer Behavioral Segmentation Portal')
st.markdown(
    'Adjust customer metrics in the sidebar below to instantly classify them'
    ' into an unsupervised K-Means behavioral segment.'
)


# Sidebar Input Controls
st.sidebar.header('Customer Metrics')
tenure = st.sidebar.slider('Tenure (Months)', 1, 60, 12, 1)
usage_frequency = st.sidebar.slider('Usage Frequency (Times/Month)', 1, 30, 10, 1)
total_spend = st.sidebar.slider('Total Spend ($)', 100.0, 1000.0, 500.0, 10.0)


# 4. Prediction & Segmentation Logic
if st.button('Identify Customer Segment', type='primary'):
  # Assemble input dataframe matching training features
  input_data = pd.DataFrame(
      [[tenure, usage_frequency, total_spend]], columns=features
  )


  # Scale input features using the saved scaler
  input_scaled = scaler.transform(input_data)


  # Predict cluster ID
  cluster_id = model.predict(input_scaled)[0]

  # Display Results
  st.subheader('Segmentation Result')
  st.success(f'**Assigned Cluster ID:** Cluster {cluster_id}')


  # Show descriptive persona
  st.markdown(cluster_descriptions.get(cluster_id))

