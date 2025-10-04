# import os
# import numpy as np
# from video_processor import VideoProcessor
# from feature_extractor import FeatureExtractor
# from query_processor import QueryProcessor
# from utils import display_video_info
# import shutil

# class VideoSummarizer:
#     def __init__(self):
#         self.video_processor = VideoProcessor()
#         self.feature_extractor = FeatureExtractor()
#         self.query_processor = QueryProcessor()
    
#     def summarize_video(self, video_path, query_image_path, 
#                        num_keyframes=20, frame_interval=5):
#         """
#         Complete video summarization pipeline
#         Args:
#             video_path: Path to input video
#             query_image_path: Path to query image
#             num_keyframes: Number of frames to include in summary
#             frame_interval: Extract one frame every N seconds
#         Returns:
#             Path to summary video
#         """
#         print("="*60)
#         print("STARTING VIDEO SUMMARIZATION PIPELINE")
#         print("="*60)
        
#         # Step 1: Display video information
#         print("\n1. VIDEO ANALYSIS:")
#         video_info = display_video_info(video_path)
#         if video_info is None:
#             return None
        
#         # Step 2: Extract frames from video
#         print(f"\n2. FRAME EXTRACTION:")
#         frame_paths = self.video_processor.extract_frames(
#             video_path, frame_interval=frame_interval
#         )
        
#         if not frame_paths:
#             print("No frames extracted. Exiting.")
#             return None
        
#         # Step 3: Extract features from all frames
#         print(f"\n3. FEATURE EXTRACTION:")
#         frame_features = self.feature_extractor.extract_features_from_frames(frame_paths)
        
#         # Step 4: Process query image
#         print(f"\n4. QUERY PROCESSING:")
#         query_features = self.query_processor.process_query_image(query_image_path)
        
#         if query_features is None:
#             print("Failed to extract features from query image. Exiting.")
#             return None
        
#         # Step 5: Calculate relevance scores
#         print(f"\n5. RELEVANCE CALCULATION:")
#         relevance_scores = self.query_processor.calculate_frame_relevance_scores(
#             query_features, frame_features
#         )
        
#         # Step 6: Rank and select top frames
#         print(f"\n6. FRAME SELECTION:")
#         ranked_frames = self.query_processor.rank_frames_by_relevance(relevance_scores)
        
#         # Select top N frames
#         selected_frames = [frame_path for frame_path, score in ranked_frames[:num_keyframes]]
        
#         # Sort selected frames chronologically
#         selected_frames.sort()
        
#         print(f"Selected {len(selected_frames)} keyframes for summary")
        
#         # Step 7: Create summary video
#         print(f"\n7. SUMMARY GENERATION:")
#         video_name = os.path.splitext(os.path.basename(video_path))[0]
#         query_name = os.path.splitext(os.path.basename(query_image_path))[0]
#         summary_path = f"results/{video_name}_{query_name}_summary.mp4"
        
#         os.makedirs("results", exist_ok=True)
#         self.video_processor.create_summary_video(selected_frames, summary_path, fps=2)
        
#         # Step 8: Copy selected keyframes to results folder
#         keyframes_dir = f"results/{video_name}_{query_name}_keyframes"
#         os.makedirs(keyframes_dir, exist_ok=True)
        
#         for i, frame_path in enumerate(selected_frames):
#             dst_path = os.path.join(keyframes_dir, f"keyframe_{i+1:03d}.jpg")
#             shutil.copy2(frame_path, dst_path)
        
#         print(f"\n8. RESULTS:")
#         print(f"  Summary video: {summary_path}")
#         print(f"  Keyframes folder: {keyframes_dir}")
#         print(f"  Total keyframes: {len(selected_frames)}")
        
#         # Calculate compression ratio
#         original_duration = video_info['duration_seconds']
#         summary_duration = len(selected_frames) / 2  # 2 fps
#         compression_ratio = (1 - summary_duration / original_duration) * 100
        
#         print(f"  Original duration: {original_duration:.2f} seconds")
#         print(f"  Summary duration: {summary_duration:.2f} seconds")
#         print(f"  Compression ratio: {compression_ratio:.1f}%")
        
#         print("\n" + "="*60)
#         print("VIDEO SUMMARIZATION COMPLETED SUCCESSFULLY!")
#         print("="*60)
        
#         return summary_path

#     def batch_summarize(self, video_folder, query_image_path, **kwargs):
#         """
#         Summarize multiple videos with the same query
#         Args:
#             video_folder: Folder containing video files
#             query_image_path: Path to query image
#             **kwargs: Additional arguments for summarize_video
#         """
#         video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
#         video_files = []
        
#         for file in os.listdir(video_folder):
#             if any(file.lower().endswith(ext) for ext in video_extensions):
#                 video_files.append(os.path.join(video_folder, file))
        
#         if not video_files:
#             print(f"No video files found in {video_folder}")
#             return
        
#         print(f"Found {len(video_files)} video files to process")
        
#         results = []
#         for i, video_path in enumerate(video_files):
#             print(f"\n{'='*20} Processing Video {i+1}/{len(video_files)} {'='*20}")
#             summary_path = self.summarize_video(video_path, query_image_path, **kwargs)
#             if summary_path:
#                 results.append(summary_path)
        
#         print(f"\nBatch processing completed. Generated {len(results)} summaries.")
#         return results





##combined motion

import os
import numpy as np
import cv2
import shutil
import json
from video_processor import VideoProcessor
from feature_extractor import FeatureExtractor, LSTMVideoSummarizer
from query_processor import QueryProcessor
from utils import display_video_info
import tensorflow as tf

class VideoSummarizer:
    def __init__(self, use_trained_model=True, model_path="models/trained_lstm_summarizer.h5"):
        self.video_processor = VideoProcessor()
        self.feature_extractor = FeatureExtractor()
        self.query_processor = QueryProcessor()
        self.use_trained_model = use_trained_model

        if use_trained_model and os.path.exists(model_path):
            print("Loading trained LSTM model...")
            self.lstm_model = LSTMVideoSummarizer()
            self.lstm_model.load_model(model_path)
            print("Trained model loaded successfully!")
        else:
            print("Trained model not found. Using query-based approach only.")
            self.lstm_model = None

    def get_lstm_importance_scores(self, frame_features):
        if self.lstm_model is None:
            return None
        try:
            if isinstance(frame_features, list):
                features_array = np.array([list(feat.values()) for feat in frame_features])
            else:
                features_array = frame_features
            importance_scores = self.lstm_model.predict_importance_scores(features_array)
            return importance_scores
        except Exception as e:
            print(f"Error getting LSTM scores: {str(e)}")
            return None

    def combine_scores(self, lstm_scores, query_scores, lstm_weight=0.6, query_weight=0.4):
        if lstm_scores is None:
            return query_scores
        lstm_scores = (lstm_scores - np.min(lstm_scores)) / (np.max(lstm_scores) - np.min(lstm_scores) + 1e-8)
        query_scores_array = np.array(list(query_scores.values()))
        query_scores_normalized = (query_scores_array - np.min(query_scores_array)) / (np.max(query_scores_array) - np.min(query_scores_array) + 1e-8)
        combined_scores = lstm_weight * lstm_scores + query_weight * query_scores_normalized
        frame_paths = list(query_scores.keys())
        combined_dict = {frame_paths[i]: combined_scores[i] for i in range(len(frame_paths))}
        return combined_dict

    def summarize_video(self, video_path, query_image_path, num_keyframes=20, frame_interval=5, use_lstm=True):
        print("="*60)
        print("STARTING ADVANCED VIDEO SUMMARIZATION PIPELINE")
        print("="*60)

        print("\n1. VIDEO ANALYSIS:")
        video_info = display_video_info(video_path)
        if video_info is None:
            return None

        print(f"\n2. FRAME EXTRACTION:")
        frame_paths = self.video_processor.extract_frames(
            video_path, frame_interval=frame_interval
        )

        if not frame_paths:
            print("No frames extracted. Exiting.")
            return None

        print(f"\n3. FEATURE EXTRACTION:")
        frame_features = self.feature_extractor.extract_features_from_frames(frame_paths)

        lstm_scores = None
        if use_lstm and self.lstm_model is not None:
            print(f"\n4. LSTM IMPORTANCE SCORING:")
            features_array = np.array([frame_features[path] for path in frame_paths])
            lstm_scores = self.get_lstm_importance_scores(features_array)
            if lstm_scores is not None:
                print(f"Generated LSTM importance scores for {len(lstm_scores)} frames")
                sorted_indices = np.argsort(lstm_scores)[::-1]
                print("Top 5 LSTM-scored frames:")
                for i, idx in enumerate(sorted_indices[:5]):
                    frame_name = os.path.basename(frame_paths[idx])
                    print(f"  {i+1}. {frame_name}: {lstm_scores[idx]:.4f}")

        print(f"\n5. QUERY PROCESSING:")
        query_features = self.query_processor.process_query_image(query_image_path)
        if query_features is None:
            print("Failed to extract features from query image. Exiting.")
            return None

        print(f"\n6. QUERY RELEVANCE CALCULATION:")
        query_relevance_scores = self.query_processor.calculate_frame_relevance_scores(
            query_features, frame_features
        )

        print(f"\n7. SCORE COMBINATION:")
        if lstm_scores is not None and use_lstm:
            final_scores = self.combine_scores(lstm_scores, query_relevance_scores)
            print("Combined LSTM and query-based scores")
            relevance_scores = final_scores
        else:
            relevance_scores = query_relevance_scores
            print("Using query-based scores only")

        print(f"\n8. FRAME SELECTION:")
        ranked_frames = self.query_processor.rank_frames_by_relevance(relevance_scores)
        selected_frames = [frame_path for frame_path, score in ranked_frames[:num_keyframes]]
        selected_frames.sort()
        print(f"Selected {len(selected_frames)} keyframes for summary")

        print(f"\n9. SUMMARY GENERATION:")
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        query_name = os.path.splitext(os.path.basename(query_image_path))[0]
        summary_path = f"results/{video_name}_{query_name}_summary.mp4"
        os.makedirs("results", exist_ok=True)
        self.video_processor.create_summary_video(selected_frames, summary_path, fps=2)

        # Step 10: Save selected keyframes and metadata
        keyframes_dir = f"results/{video_name}_{query_name}_keyframes"
        os.makedirs(keyframes_dir, exist_ok=True)
        metadata = {
            'video_path': video_path,
            'query_path': query_image_path,
            'num_keyframes': len(selected_frames),
            'frame_interval': frame_interval,
            'used_lstm': use_lstm and lstm_scores is not None,
            'selected_frames': selected_frames,
            'top_scores': [(path, float(score)) for path, score in ranked_frames[:10]]
        }
        for i, frame_path in enumerate(selected_frames):
            dst_path = os.path.join(keyframes_dir, f"keyframe_{i+1:03d}.jpg")
            shutil.copy2(frame_path, dst_path)
        metadata_file = os.path.join(keyframes_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n10. RESULTS:")
        print(f"  Summary video: {summary_path}")
        print(f"  Keyframes folder: {keyframes_dir}")
        print(f"  Total keyframes: {len(selected_frames)}")
        print(f"  Used LSTM model: {use_lstm and lstm_scores is not None}")

        original_duration = video_info['duration_seconds']
        summary_duration = len(selected_frames) / 2  # 2 fps
        compression_ratio = (1 - summary_duration / original_duration) * 100

        print(f"  Original duration: {original_duration:.2f} seconds")
        print(f"  Summary duration: {summary_duration:.2f} seconds")
        print(f"  Compression ratio: {compression_ratio:.1f}%")

        print("\n" + "="*60)
        print("ADVANCED VIDEO SUMMARIZATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        return summary_path

    # Motion summary feature (added)
    def summarize_motion(self, video_path, query_image_path=None, threshold=30, min_segment_len=20):
        segments = self.video_processor.extract_motion_segments(video_path, threshold, min_segment_len)
        if not segments:
            print("No motion segments detected.")
            return None
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        summary_path = f"results/{video_name}_motion_summary.mp4"
        os.makedirs("results", exist_ok=True)
        self.video_processor.save_motion_summary(video_path, segments, summary_path)
        print(f"Motion summary saved to: {summary_path}")
        return summary_path

    # Dispatcher for UI selection
    def summarize(self, video_path, query_image_path, summary_type="Keyframe", **kwargs):
        if summary_type == "Keyframe":
            return self.summarize_video(video_path, query_image_path, **kwargs)
        elif summary_type == "Motion":
            return self.summarize_motion(video_path, query_image_path,
                                         threshold=kwargs.get("motion_threshold", 30),
                                         min_segment_len=kwargs.get("min_segment_len", 20))
        else:
            raise ValueError("Invalid summary type")

    def batch_summarize(self, video_folder, query_image_path, **kwargs):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        video_files = []
        for file in os.listdir(video_folder):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(video_folder, file))
        if not video_files:
            print(f"No video files found in {video_folder}")
            return
        print(f"Found {len(video_files)} video files to process")
        results = []
        for i, video_path in enumerate(video_files):
            print(f"\n{'='*20} Processing Video {i+1}/{len(video_files)} {'='*20}")
            summary_path = self.summarize_video(video_path, query_image_path, **kwargs)
            if summary_path:
                results.append(summary_path)
        print(f"\nBatch processing completed. Generated {len(results)} summaries.")
        return results
