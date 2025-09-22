# import streamlit as st
# from summarizer import VideoSummarizer
# import tempfile
# import os

# def main():
#     st.title("AI Video Summarization Tool")
    
#     uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
#     uploaded_query = st.file_uploader("Choose a query image", type=['jpg', 'png'])
    
#     if uploaded_video and uploaded_query:
#         if st.button("Generate Summary"):
#             summarizer = VideoSummarizer()
            
#             # Save uploaded files temporarily
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
#                 tmp_video.write(uploaded_video.read())
#                 video_path = tmp_video.name
            
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_query:
#                 tmp_query.write(uploaded_query.read())
#                 query_path = tmp_query.name
            
#             # Generate summary
#             try:
#                 summary_path = summarizer.summarize_video(
#                     video_path=video_path,
#                     query_image_path=query_path,
#                     num_keyframes=15,
#                     frame_interval=3
#                 )
#                 st.success("Summary generated successfully!")
#                 st.video(summary_path)  # Display the summarized video
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

# if __name__ == "__main__":
#     main()



import streamlit as st
from summarizer import VideoSummarizer
import tempfile
import os

def main():
    st.title("AI Video Summarization Tool")
    
    uploaded_video = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    uploaded_query = st.file_uploader("Choose a query image", type=['jpg', 'png'])
    
    if uploaded_video and uploaded_query:
        if st.button("Generate Summary"):
            summarizer = VideoSummarizer()
            
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                tmp_video.write(uploaded_video.read())
                video_path = tmp_video.name
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_query:
                tmp_query.write(uploaded_query.read())
                query_path = tmp_query.name
            
            # Generate summary
            try:
                summary_path = summarizer.summarize_video(
                    video_path=video_path,
                    query_image_path=query_path,
                    num_keyframes=15,
                    frame_interval=3
                )
                
                if summary_path and os.path.exists(summary_path):
                    st.success("Summary generated successfully!")
                    st.write("### Preview of Summary Video:")
                    
                    # Open video as bytes and feed to Streamlit
                    with open(summary_path, "rb") as f:
                        video_bytes = f.read()
                    st.video(video_bytes)
                    
                    st.write(f"✅ Summary saved at: `{summary_path}`")
                else:
                    st.error("Summary video not found or failed to generate.")
                    
            except Exception as e:
                st.error(f"Error during summarization: {str(e)}")

if __name__ == "__main__":
    main()
