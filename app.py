import streamlit as st
import torch
from sentence_transformers import util
import pickle
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
import boto3
import os

# AWS S3 Bucket Details
bucket_name = 'ml-proj'  # Name of the S3 bucket
s3_folder = 'models/'  # Folder name in the S3 bucket

# AWS S3 Client initialized with the EC2 instance's IAM role
s3 = boto3.client('s3', region_name='us-west-2')

# Function to download a file from S3 to the local file system
def download_from_s3(s3_key, local_path):
    s3.download_file(bucket_name, s3_key, local_path)

# Local paths for the downloaded model files
local_model_paths = {
    'embeddings': '/tmp/embeddings.pkl',
    'sentences': '/tmp/sentences.pkl',
    'rec_model': '/tmp/rec_model.pkl',
    'model': '/tmp/model.h5',
    'text_vectorizer_config': '/tmp/text_vectorizer_config.pkl',
    'text_vectorizer_weights': '/tmp/text_vectorizer_weights.pkl',
    'vocab': '/tmp/vocab.pkl'
}

# Download model files from S3 bucket
for model_name, local_path in local_model_paths.items():
    s3_key = f'{s3_folder}{model_name}'  # S3 Key based on the folder and file name
    download_from_s3(s3_key, local_path)

# Load saved recommendation models
embeddings = pickle.load(open(local_model_paths['embeddings'], 'rb'))
sentences = pickle.load(open(local_model_paths['sentences'], 'rb'))
rec_model = pickle.load(open(local_model_paths['rec_model'], 'rb'))

# Load saved prediction models
loaded_model = keras.models.load_model(local_model_paths['model'])
with open(local_model_paths['text_vectorizer_config'], "rb") as f:
    saved_text_vectorizer_config = pickle.load(f)
loaded_text_vectorizer = TextVectorization.from_config(saved_text_vectorizer_config)
with open(local_model_paths['text_vectorizer_weights'], "rb") as f:
    weights = pickle.load(f)
    loaded_text_vectorizer.set_weights(weights)
with open(local_model_paths['vocab'], "rb") as f:
    loaded_vocab = pickle.load(f)

# Define custom functions
def recommendation(input_paper):
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
    papers_list = [sentences[i.item()] for i in top_similar_papers.indices]
    return papers_list

def invert_multi_hot(encoded_labels):
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(loaded_vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_lookup):
    preprocessed_abstract = vectorizer([abstract])
    predictions = model.predict(preprocessed_abstract)
    predicted_labels = label_lookup(np.round(predictions).astype(int)[0])
    return predicted_labels

# Create the Streamlit app
st.title('Research Papers Recommendation and Subject Area Prediction App')
st.markdown("## LLM and Deep Learning Based App")

num_recommendations = st.sidebar.selectbox('Number of recommendations:', (3, 5, 10), index=1)
input_paper = st.text_input("Enter paper title:")
new_abstract = st.text_area("Paste paper abstract:")

if st.button("Analyze"):
    with st.spinner('Generating recommendations and predictions...'):
        recommend_papers = recommendation(input_paper)[:num_recommendations]
        predicted_categories = predict_category(new_abstract, loaded_model, loaded_text_vectorizer, invert_multi_hot)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recommended Papers")
        for paper in recommend_papers:
            st.write(paper)

    with col2:
        st.subheader("Predicted Subject Area")
        st.write(", ".join(predicted_categories))
