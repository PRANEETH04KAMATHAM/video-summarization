
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
        
        first_frame = cv2.imread(selected_frames[0])
        height, width, layers = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Creating summary video with {len(selected_frames)} frames...")
        for frame_path in selected_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
        out.release()
        print(f"Summary video saved to: {output_path}")

    # ---- MOTION FEATURE ----
    def extract_motion_segments(self, video_path, threshold=30, min_segment_len=20):
        """
        Extract segments of video with significant motion.
        Args:
            video_path: Path to input video
            threshold: Motion detection threshold
            min_segment_len: Minimum segment length in frames
        Returns:
            List of (start_frame, end_frame) tuples
        """
        cap = cv2.VideoCapture(video_path)
        last_gray = None
        segments = []
        start = None
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if last_gray is not None:
                diff = cv2.absdiff(gray, last_gray)
                score = np.mean(diff)
                if score > threshold:
                    if start is None:
                        start = frame_idx
                else:
                    if start is not None and frame_idx - start >= min_segment_len:
                        segments.append((start, frame_idx))
                    start = None
            last_gray = gray
            frame_idx += 1
        cap.release()
        return segments

    def save_motion_summary(self, video_path, segments, output_path):
        """
        Save video summary as concatenation of motion segments.
        Args:
            video_path: Path to input video
            segments: List of (start_frame, end_frame) tuples
            output_path: Path for output summary video
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for start, end in segments:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for i in range(start, end):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
        out.release()
        cap.release()
        print(f"Motion summary saved to: {output_path}")
