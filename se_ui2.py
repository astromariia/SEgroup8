"""
se_ui2.py

A simple Gradio application for running YOLO-based object detection on
images and videos. This version is simplified and heavily commented
for the purpose of generating clean documentation with pydoc.
"""

import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile
import os


# ---------------------------------------------------------------------------
# Load YOLO Model
# ---------------------------------------------------------------------------
# The YOLO model is loaded once when the script starts. The file "best.pt"
# must be present in the working directory.
model = YOLO("best.pt")


# ---------------------------------------------------------------------------
# Image Processing Function
# ---------------------------------------------------------------------------
def process_image(image, confidence):
    """
    Perform object detection on a single image.

    Parameters
    ----------
    image : numpy.ndarray
        The input image uploaded through Gradio (RGB format).
    confidence : float
        Confidence threshold for YOLO detections (0.0 - 1.0).

    Returns
    -------
    numpy.ndarray
        The annotated image with bounding boxes drawn.
    """
    # Run YOLO inference
    results = model(image, conf=confidence)

    # Draw bounding boxes and labels on the image
    annotated = results[0].plot()

    # YOLO outputs BGR images; convert to RGB for Gradio display
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return annotated


# ---------------------------------------------------------------------------
# Video Processing Function
# ---------------------------------------------------------------------------
def process_video(video_path, confidence, progress=gr.Progress()):
    """
    Perform object detection on each frame of an uploaded video.

    Parameters
    ----------
    video_path : str
        Path to the uploaded video file.
    confidence : float
        YOLO detection confidence threshold.
    progress : gr.Progress
        Gradio progress bar (injected automatically).

    Returns
    -------
    str or None
        Path to the processed video file, or None if processing failed.
    """
    if video_path is None:
        return None

    # Try to open the uploaded video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Fetch video metadata
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a temporary output video file
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_output.name
    temp_output.close()

    # Attempt several codecs for compatibility
    codecs = ["mp4v", "avc1", "H264"]
    out = None

    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            break
        out.release()

    if not out or not out.isOpened():
        cap.release()
        return None

    frame_idx = 0

    # Process frames one by one
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference on each frame
        results = model(frame, conf=confidence)
        annotated = results[0].plot()

        # Write annotated frame to output video
        out.write(annotated)

        frame_idx += 1

        # Update progress bar
        if total_frames > 0:
            progress(frame_idx / total_frames,
                     desc=f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    out.release()

    # Validate that the video file was actually written
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return output_path

    os.remove(output_path)
    return None


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="YOLO Object Detection") as demo:

    gr.Markdown("## Object Detection Demo")

    # -------------------------- Image Tab --------------------------
    with gr.Tab("Image Detection"):
        image_input = gr.Image(type="numpy", label="Upload Image")
        image_conf = gr.Slider(0, 1, value=0.25, label="Confidence Threshold")
        image_button = gr.Button("Detect Objects")
        image_output = gr.Image(label="Annotated Image")

        image_button.click(
            fn=process_image,
            inputs=[image_input, image_conf],
            outputs=image_output
        )

    # -------------------------- Video Tab --------------------------
    with gr.Tab("Video Detection"):
        video_input = gr.Video(label="Upload Video")
        video_conf = gr.Slider(0, 1, value=0.25, label="Confidence Threshold")
        video_button = gr.Button("Process Video")
        video_output = gr.Video(label="Processed Video")

        video_button.click(
            fn=process_video,
            inputs=[video_input, video_conf],
            outputs=video_output
        )


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Launches the app with an optional shareable public link
    demo.launch(share=True)
