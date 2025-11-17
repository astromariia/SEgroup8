import gradio as gr
import cv2
from ultralytics import YOLO
import tempfile
import os

model = YOLO('best.pt')

def process_image(image, confidence):
    results = model(image, conf=confidence)
    annotated_image = results[0].plot()
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image

def process_video(video_path, confidence, progress=gr.Progress()):
    if video_path is None:
        return None
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = temp_output.name
    temp_output.close()

    codecs = ['avc1', 'H264', 'X264', 'mp4v']
    out = None
    
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if out.isOpened():
            print(f"Using codec: {codec}")
            break
        out.release()
    
    if not out.isOpened():
        cap.release()
        print("Failed to open video writer with any codec")
        return None
    
    frame_count = 0
    processed_frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break

        results = model(frame, conf=confidence)
        annotated_frame = results[0].plot()
        
        out.write(annotated_frame)
        
        frame_count += 1
        if total_frames > 0:
            progress((frame_count / total_frames), desc=f"Processing frame {frame_count}/{total_frames}")
    
    cap.release()
    out.release()
    
    # Check if file was created and has content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"Video saved successfully: {output_path}, size: {os.path.getsize(output_path)} bytes")
        return output_path
    else:
        print(f"Video file creation failed or file is too small")
        if os.path.exists(output_path):
            os.remove(output_path)
        return None

with gr.Blocks(title="Software Engineering Detection") as demo:
    gr.Markdown("Project Object Detection")
    gr.Markdown("Upload an image or video for object detection.")
    
    with gr.Tab("Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="numpy", label="Upload Image")
                image_confidence = gr.Slider(0, 1, value=0.25, label="Confidence Threshold")
                image_button = gr.Button("Detect Objects", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(label="Detection Results")
        
        image_button.click(
            fn=process_image,
            inputs=[image_input, image_confidence],
            outputs=image_output
        )
    
    with gr.Tab("Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                video_confidence = gr.Slider(0, 1, value=0.25, label="Confidence Threshold")
                video_button = gr.Button("Process Video", variant="primary")
                gr.Markdown("Video processing may take a while depending on length")
            
            with gr.Column():
                video_output = gr.Video(label="Processed Video")
        
        video_button.click(
            fn=process_video,
            inputs=[video_input, video_confidence],
            outputs=video_output
        )

if __name__ == "__main__":
    demo.launch(share=True)