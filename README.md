# Video Summarization Project

This project implements a query-based video summarization system using deep learning models including ResNet50 and LSTM. It includes frame extraction, feature extraction, temporal scoring, and query relevance matching.

## Features
- Frame extraction from videos using OpenCV
- Visual feature extraction with pre-trained ResNet50
- Temporal modeling with Bidirectional LSTM
- Query-driven summarization with cosine similarity
- User interface via Streamlit

## How to Run
1. Setup environment: `pip install -r requirements.txt`
2. Run training: `python train_model.py --prepare_data`
3. Run summarization: `python main.py`

## Dependencies
- tensorflow
- numpy
- opencv-python
- matplotlib
- streamlit

