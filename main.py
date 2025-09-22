# #!/usr/bin/env python3
# """
# AI-Driven Video Summarization
# Main application file for query-based video summarization
# """

# import os
# import sys
# from summarizer import VideoSummarizer
# from utils import create_directories, display_video_info

# def main():
#     print("AI-Driven Video Summarization System")
#     print("=====================================")
    
#     # Create necessary directories
#     create_directories()
    
#     # Initialize summarizer
#     summarizer = VideoSummarizer()
    
#     # Example usage - you can modify these paths
#     video_path = input("Enter path to video file (or press Enter for sample): ").strip()
#     if not video_path:
#         video_path = "data/videos/sample_video_1.mp4"
    
#     query_image_path = input("Enter path to query image (or press Enter for sample): ").strip()
#     if not query_image_path:
#         query_image_path = "query_images/sample_query.jpg"
    
#     # Check if files exist
#     if not os.path.exists(video_path):
#         print(f"Error: Video file not found: {video_path}")
#         print("Please place a video file in the data/videos/ folder")
#         return
    
#     if not os.path.exists(query_image_path):
#         print(f"Error: Query image not found: {query_image_path}")
#         print("Please place a query image in the query_images/ folder")
#         return
    
#     # Get user preferences
#     try:
#         num_keyframes = int(input("Enter number of keyframes for summary (default 20): ") or "20")
#         frame_interval = int(input("Enter frame extraction interval in seconds (default 5): ") or "5")
#     except ValueError:
#         num_keyframes = 20
#         frame_interval = 5
    
#     # Run summarization
#     try:
#         summary_path = summarizer.summarize_video(
#             video_path=video_path,
#             query_image_path=query_image_path,
#             num_keyframes=num_keyframes,
#             frame_interval=frame_interval
#         )
        
#         if summary_path:
#             print(f"\nSummarization completed successfully!")
#             print(f"Check the results folder for your summary video and keyframes.")
#         else:
#             print("Summarization failed. Please check the error messages above.")
            
#     except Exception as e:
#         print(f"Error during summarization: {str(e)}")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
AI-Driven Video Summarization
Main application file for query-based video summarization with trained model
"""

import os
import sys
import argparse
from summarizer import VideoSummarizer
from utils import create_directories, display_video_info

def main():
    parser = argparse.ArgumentParser(description='AI-Driven Video Summarization')
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--query', type=str, help='Path to query image')
    parser.add_argument('--keyframes', type=int, default=20, help='Number of keyframes')
    parser.add_argument('--interval', type=int, default=5, help='Frame extraction interval (seconds)')
    parser.add_argument('--no-lstm', action='store_true', help='Disable LSTM model')
    parser.add_argument('--model', type=str, default='models/trained_lstm_summarizer.h5', help='Path to trained model')
    args = parser.parse_args()
    
    print("AI-Driven Video Summarization System")
    print("====================================")
    
    # Create necessary directories
    create_directories()
    
    # Check if trained model exists
    use_trained_model = not args.no_lstm and os.path.exists(args.model)
    if use_trained_model:
        print(f"✓ Using trained LSTM model: {args.model}")
    else:
        if not args.no_lstm:
            print(f"⚠ Trained model not found: {args.model}")
            print("  Run 'python train_model.py --prepare_data' to train a model first")
        print("  Using query-based approach only")
    
    # Initialize summarizer
    summarizer = VideoSummarizer(use_trained_model=use_trained_model, model_path=args.model)
    
    # Get input paths
    video_path = args.video or input("Enter path to video file (or press Enter for sample): ").strip()
    if not video_path:
        video_path = "data/videos/sample_video_1.mp4"
    
    query_image_path = args.query or input("Enter path to query image (or press Enter for sample): ").strip()
    if not query_image_path:
        query_image_path = "query_images/sample_query.jpg"
    
    # Check if files exist
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Please place a video file in the data/videos/ folder")
        return
    
    if not os.path.exists(query_image_path):
        print(f"Error: Query image not found: {query_image_path}")
        print("Please place a query image in the query_images/ folder")
        return
    
    # Run summarization
    try:
        summary_path = summarizer.summarize_video(
            video_path=video_path,
            query_image_path=query_image_path,
            num_keyframes=args.keyframes,
            frame_interval=args.interval,
            use_lstm=use_trained_model
        )
        
        if summary_path:
            print(f"\nSummarization completed successfully!")
            print(f"Check the results folder for your summary video and keyframes.")
        else:
            print("Summarization failed. Please check the error messages above.")
            
    except Exception as e:
        print(f"Error during summarization: {str(e)}")

if __name__ == "__main__":
    main()
