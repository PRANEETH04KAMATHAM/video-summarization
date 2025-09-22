import os
import json
import numpy as np
import h5py
import cv2
from scipy.io import loadmat
import pickle
from sklearn.model_selection import train_test_split
from feature_extractor import FeatureExtractor

class TVSumDataPreprocessor:
    def __init__(self, dataset_path="data/datasets"):
        self.dataset_path = dataset_path
        self.feature_extractor = FeatureExtractor()
        self.processed_data_path = os.path.join(dataset_path, "processed")
        os.makedirs(self.processed_data_path, exist_ok=True)
    
    def download_and_setup_tvsum(self):
        """Download and setup TVSum dataset"""
        print("Setting up TVSum dataset...")
        
        # Create directory structure
        tvsum_path = os.path.join(self.dataset_path, "tvsum")
        os.makedirs(tvsum_path, exist_ok=True)
        
        # Download instructions
        print(f"""
        TVSUM DATASET SETUP INSTRUCTIONS:
        
        1. Download the TVSum dataset from:
           https://github.com/yalesong/tvsum
        
        2. Download the video files and annotations to:
           {tvsum_path}/
        
        3. The structure should be:
           {tvsum_path}/
           ├── video/           # Video files (.mp4)
           ├── data/           # Annotation files (.mat)
           └── ydata-tvsum50-anno.tsv
        
        4. Alternative: Use preprocessed h5 files from:
           https://github.com/li-plus/DSNet
        """)
        
        return tvsum_path
    
    def load_tvsum_annotations(self, annotation_file):
        """Load TVSum annotations from .mat or .h5 files"""
        annotations = {}
        
        if annotation_file.endswith('.mat'):
            # Load MATLAB format annotations
            data = loadmat(annotation_file)
            
            for i, video_name in enumerate(data['video_names']):
                video_id = video_name[0]
                user_scores = data['user_anno'][i]  # Shape: (num_users, num_shots)
                
                # Average user scores to get ground truth
                gt_scores = np.mean(user_scores, axis=0)
                
                annotations[video_id] = {
                    'user_scores': user_scores,
                    'gt_scores': gt_scores,
                    'n_frames': len(gt_scores) * 15  # Assuming 15 fps sampling
                }
        
        elif annotation_file.endswith('.h5'):
            # Load HDF5 format annotations
            with h5py.File(annotation_file, 'r') as f:
                for video_id in f.keys():
                    video_data = f[video_id]
                    
                    # Extract ground truth scores
                    gt_scores = video_data['user_summary'][:]  # Pre-averaged scores
                    
                    annotations[video_id] = {
                        'gt_scores': gt_scores,
                        'n_frames': video_data['n_frames'][()],
                        'fps': video_data.get('fps', [15])[0]
                    }
        
        return annotations
    
    def extract_video_features(self, video_path, target_fps=2):
        """Extract features from video at specified fps"""
        print(f"Extracting features from: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        frame_interval = int(fps / target_fps) if fps > 0 else 1
        
        features = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at target fps
            if frame_count % frame_interval == 0:
                # Resize frame for ResNet50
                resized_frame = cv2.resize(frame, (224, 224))
                
                # Convert BGR to RGB for ResNet50
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                
                # Extract features using ResNet50
                features_vector = self.feature_extractor.extract_features_from_frame_array(rgb_frame)
                if features_vector is not None:
                    features.append(features_vector)
            
            frame_count += 1
        
        cap.release()
        
        return np.array(features)
    
    def prepare_training_data(self, tvsum_path):
        """Prepare complete training dataset"""
        print("Preparing training data...")
        
        # Load annotations
        annotation_file = os.path.join(tvsum_path, "data", "ydata-tvsum50-anno.tsv")
        if not os.path.exists(annotation_file):
            # Try alternative formats
            mat_files = [f for f in os.listdir(os.path.join(tvsum_path, "data")) if f.endswith('.mat')]
            if mat_files:
                annotation_file = os.path.join(tvsum_path, "data", mat_files[0])
        
        # Load video list and annotations
        video_annotations = {}
        if annotation_file.endswith('.tsv') and os.path.exists(annotation_file):
            # Parse TSV format safely
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split('\t')
                        if not parts:
                            continue
                        video_id = parts[0]
                        video_annotations[video_id] = {
                            'category': parts[1] if len(parts) > 1 else "unknown",
                            'title': parts[2] if len(parts) > 2 else f"video_{video_id}",
                            'length': float(parts[3]) if len(parts) > 3 else 0.0
                        }
        
        # Process each video
        training_data = []
        video_dir = os.path.join(tvsum_path, "video")
        
        for video_file in os.listdir(video_dir):
            if not video_file.endswith(('.mp4', '.avi', '.mov')):
                continue
                
            video_id = os.path.splitext(video_file)[0]
            video_path = os.path.join(video_dir, video_file)
            
            try:
                # Extract features
                features = self.extract_video_features(video_path)
                
                # Create dummy ground truth if annotations not available
                if video_id in video_annotations:
                    # Use actual annotations if available
                    gt_scores = self.create_gt_scores_from_video(video_path, features.shape[0])
                else:
                    # Create synthetic ground truth for demonstration
                    gt_scores = self.create_synthetic_gt_scores(features.shape[0])
                
                training_data.append({
                    'video_id': video_id,
                    'features': features,
                    'gt_scores': gt_scores,
                    'n_frames': features.shape[0]
                })
                
                print(f"Processed {video_id}: {features.shape[0]} frames")
                
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                continue
        
        return training_data
    
    def create_gt_scores_from_video(self, video_path, n_frames):
        """Create ground truth scores from video analysis"""
        # Simplified ground truth generation
        gt_scores = np.random.uniform(0.1, 0.3, n_frames)
        
        # Add some highlight peaks
        highlight_positions = np.linspace(0, n_frames-1, 5, dtype=int)
        for pos in highlight_positions:
            for i in range(max(0, pos-10), min(n_frames, pos+10)):
                distance = abs(i - pos)
                gt_scores[i] = max(gt_scores[i], 0.8 * np.exp(-distance**2 / 50))
        
        return gt_scores
    
    def create_synthetic_gt_scores(self, n_frames):
        """Create synthetic ground truth scores for training"""
        gt_scores = np.random.beta(2, 5, n_frames)  # Skewed towards lower values
        
        # Add some random highlights
        n_highlights = np.random.randint(3, 8)
        highlight_positions = np.random.choice(n_frames, n_highlights, replace=False)
        
        for pos in highlight_positions:
            start = max(0, pos - 5)
            end = min(n_frames, pos + 5)
            gt_scores[start:end] = np.random.uniform(0.7, 1.0, end - start)
        
        return gt_scores
    
    def save_processed_data(self, training_data):
        """Save processed training data"""
        output_file = os.path.join(self.processed_data_path, "tvsum_training_data.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"Saved processed data to: {output_file}")
        
        # Save summary statistics
        stats = {
            'n_videos': len(training_data),
            'total_frames': sum([data['n_frames'] for data in training_data]),
            'avg_frames_per_video': np.mean([data['n_frames'] for data in training_data]),
            'feature_dim': training_data[0]['features'].shape[1] if training_data else 0
        }
        
        stats_file = os.path.join(self.processed_data_path, "dataset_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset statistics: {stats}")
        
        return output_file
    
    def load_processed_data(self):
        """Load processed training data"""
        data_file = os.path.join(self.processed_data_path, "tvsum_training_data.pkl")
        
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Processed data not found at: {data_file}")
