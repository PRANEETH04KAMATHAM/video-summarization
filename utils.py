import os
import numpy as np
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def create_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['data/videos', 'data/frames', 'data/datasets', 'models', 'results', 'query_images']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("Directories created successfully!")

def display_video_info(video_path):
    """Display basic information about a video file"""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    info = {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration_seconds': duration,
        'duration_minutes': duration / 60
    }
    
    print(f"Video Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frame Count: {frame_count}")
    print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    cap.release()
    return info

def calculate_cosine_similarity(features1, features2):
    """Calculate cosine similarity between two feature vectors"""
    # Ensure features are 2D arrays
    if features1.ndim == 1:
        features1 = features1.reshape(1, -1)
    if features2.ndim == 1:
        features2 = features2.reshape(1, -1)
    
    return cosine_similarity(features1, features2)[0][0]
