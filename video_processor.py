import cv2
import os
import numpy as np
from utils import create_directories

class VideoProcessor:
    def __init__(self, output_dir="data/frames"):
        self.output_dir = output_dir
        create_directories()
    
    def extract_frames(self, video_path, frame_interval=5):
        """
        Extract frames from video at specified intervals
        Args:
            video_path: Path to input video
            frame_interval: Extract one frame every N seconds
        Returns:
            List of frame file paths
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Create output directory for this video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frame_output_dir = os.path.join(self.output_dir, video_name)
        os.makedirs(frame_output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_frames = int(fps * frame_interval)  # Convert seconds to frames
        
        frame_count = 0
        saved_frames = []
        
        print(f"Extracting frames from {video_name}...")
        print(f"FPS: {fps}, Frame interval: {frame_interval} seconds ({frame_interval_frames} frames)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at specified intervals
            if frame_count % frame_interval_frames == 0:
                frame_filename = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(frame_output_dir, frame_filename)
                
                # Resize frame to standard size (224x224 for ResNet50)
                resized_frame = cv2.resize(frame, (224, 224))
                cv2.imwrite(frame_path, resized_frame)
                saved_frames.append(frame_path)
                
                if len(saved_frames) % 50 == 0:
                    print(f"  Extracted {len(saved_frames)} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"Total frames extracted: {len(saved_frames)}")
        return saved_frames
    
    def create_summary_video(self, selected_frames, output_path, fps=2):
        """
        Create summary video from selected frames
        Args:
            selected_frames: List of frame file paths
            output_path: Path for output summary video
            fps: Frames per second for output video
        """
        if not selected_frames:
            print("No frames provided for summary creation")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(selected_frames[0])
        height, width, layers = first_frame.shape
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating summary video with {len(selected_frames)} frames...")
        
        for frame_path in selected_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
        
        out.release()
        print(f"Summary video saved to: {output_path}")
