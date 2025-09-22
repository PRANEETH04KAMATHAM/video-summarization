import os
import time
import numpy as np
from summarizer import VideoSummarizer
from utils import display_video_info

def evaluate_performance():
    """Evaluate summarization performance on multiple test cases"""
    
    # Test cases: (video_path, query_path, expected_description)
    test_cases = [
        ("data/videos/sample_video_1.mp4", "query_images/sample_query.png", "Person detection"),

        # Add more test cases as needed
    ]
    
    summarizer = VideoSummarizer()
    results = []
    
    print("PERFORMANCE EVALUATION")
    print("=" * 50)
    
    for i, (video_path, query_path, description) in enumerate(test_cases):
        if not os.path.exists(video_path) or not os.path.exists(query_path):
            print(f"Skipping test {i+1}: Files not found")
            continue
        
        print(f"\nTest {i+1}: {description}")
        print("-" * 30)
        
        start_time = time.time()
        
        try:
            summary_path = summarizer.summarize_video(
                video_path=video_path,
                query_image_path=query_path,
                num_keyframes=15,
                frame_interval=3
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            if summary_path:
                # Get video info for metrics
                video_info = display_video_info(video_path)
                original_duration = video_info['duration_seconds']
                summary_duration = 15 / 2  # 15 frames at 2 fps
                compression_ratio = (1 - summary_duration / original_duration) * 100
                
                result = {
                    'test_name': description,
                    'video_path': video_path,
                    'processing_time': processing_time,
                    'original_duration': original_duration,
                    'summary_duration': summary_duration,
                    'compression_ratio': compression_ratio,
                    'success': True
                }
                
                print(f"✓ Success - Processing time: {processing_time:.2f}s")
                print(f"  Compression ratio: {compression_ratio:.1f}%")
            else:
                result = {
                    'test_name': description,
                    'success': False,
                    'processing_time': processing_time
                }
                print(f"✗ Failed")
            
            results.append(result)
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({
                'test_name': description,
                'success': False,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if successful_tests:
        avg_processing_time = np.mean([r['processing_time'] for r in successful_tests])
        avg_compression_ratio = np.mean([r['compression_ratio'] for r in successful_tests])
        
        print(f"\nAverage processing time: {avg_processing_time:.2f} seconds")
        print(f"Average compression ratio: {avg_compression_ratio:.1f}%")

if __name__ == "__main__":
    evaluate_performance()
