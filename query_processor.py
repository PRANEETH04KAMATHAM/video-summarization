import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from feature_extractor import FeatureExtractor
import os

class QueryProcessor:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def process_query_image(self, query_image_path):
        """
        Process query image and extract features
        Args:
            query_image_path: Path to query image
        Returns:
            Query features
        """
        if not os.path.exists(query_image_path):
            raise FileNotFoundError(f"Query image not found: {query_image_path}")
        
        return self.feature_extractor.extract_features_from_query_image(query_image_path)
    
    def calculate_frame_relevance_scores(self, query_features, frame_features_dict):
        """
        Calculate relevance scores between query and all frames
        Args:
            query_features: Features from query image
            frame_features_dict: Dictionary mapping frame paths to features
        Returns:
            Dictionary mapping frame paths to relevance scores
        """
        relevance_scores = {}
        query_features = query_features.reshape(1, -1)  # Ensure 2D shape
        
        print("Calculating relevance scores...")
        
        for frame_path, frame_features in frame_features_dict.items():
            frame_features = frame_features.reshape(1, -1)  # Ensure 2D shape
            
            # Calculate cosine similarity
            similarity = cosine_similarity(query_features, frame_features)[0][0]
            relevance_scores[frame_path] = similarity
        
        print(f"Calculated relevance scores for {len(relevance_scores)} frames")
        return relevance_scores
    
    def rank_frames_by_relevance(self, relevance_scores):
        """
        Rank frames by their relevance scores
        Args:
            relevance_scores: Dictionary of frame paths to scores
        Returns:
            List of tuples (frame_path, score) sorted by score (descending)
        """
        ranked_frames = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
        
        print(f"Top 5 most relevant frames:")
        for i, (frame_path, score) in enumerate(ranked_frames[:5]):
            frame_name = os.path.basename(frame_path)
            print(f"  {i+1}. {frame_name}: {score:.4f}")
        
        return ranked_frames
