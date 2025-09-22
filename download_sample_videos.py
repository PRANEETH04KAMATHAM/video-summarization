import requests
import os

def download_sample_videos():
    urls = [
        "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_1mb.mp4",
        "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_5mb.mp4",
        "https://sample-videos.com/video123/mp4/720/big_buck_bunny_720p_10mb.mp4",
    ]

    output_path = "data/videos/"
    os.makedirs(output_path, exist_ok=True)

    for i, url in enumerate(urls):
        filename = f"sample_video_{i+1}.mp4"
        filepath = os.path.join(output_path, filename)
        try:
            print(f"Downloading {filename}...")
            r = requests.get(url, stream=True)
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")

if __name__ == "__main__":
    download_sample_videos()

