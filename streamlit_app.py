
import streamlit as st
from summarizer import VideoSummarizer
import tempfile, os

def main():
    st.title("AI Video Summarization Tool")
    uploadedvideo = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    uploadedquery = st.file_uploader("Choose a query image", type=["jpg", "png"])

    if uploadedvideo and uploadedquery:
        if st.button("Generate Summary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpvideo:
                tmpvideo.write(uploadedvideo.read())
                videopath = tmpvideo.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmpquery:
                tmpquery.write(uploadedquery.read())
                querypath = tmpquery.name
            try:
                summarizer = VideoSummarizer()
                # Use correct method and argument names
                summarypath = summarizer.summarize_video(
                    video_path=videopath,
                    query_image_path=querypath,
                    num_keyframes=15,
                    frame_interval=3,
                    use_lstm=True
                )
                if summarypath and os.path.exists(summarypath):
                    st.success("Summary generated successfully!")
                    st.write(f"Summary saved at `{summarypath}`")
                    with open(summarypath, "rb") as f:
                        videobytes = f.read()
                    st.video(videobytes)
                    st.download_button(
                        label="Download Summary Video",
                        data=videobytes,
                        file_name=os.path.basename(summarypath),
                        mime="video/mp4"
                    )
                else:
                    st.error("Summary video not found or failed to generate.")
            except Exception as e:
                st.error(f"Error during summarization: {e}")

if __name__ == "__main__":
    main()
