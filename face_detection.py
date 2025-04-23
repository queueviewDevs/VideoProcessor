import cv2
import numpy as np
import os
import subprocess
import threading
import queue
import time
import shutil

# Load DNN model
neural_network = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "./res10_300x300_ssd_iter_140000.caffemodel")

# Load face covers
face_covers = [cv2.imread(os.path.join("./smileys/", f), cv2.IMREAD_UNCHANGED)
               for f in sorted(os.listdir("./smileys/")) if f.endswith(".png")]

# Use OpenCL if available (may help on AMD)
cv2.ocl.setUseOpenCL(True)

# Thread-safe frame queue
frame_queue = queue.Queue(maxsize=10)

def add_face_cover(background, cover, x, y, w, h, scale=1.5):
    new_w = int(w * scale)
    new_h = int(h * scale)
    cover_resized = cv2.resize(cover, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x -= (new_w - w) // 2
    y -= (new_h - h) // 2
    x, y = max(0, x), max(0, y)
    end_x, end_y = min(background.shape[1], x + new_w), min(background.shape[0], y + new_h)
    cover_resized = cover_resized[:end_y - y, :end_x - x]
    if cover_resized.shape[2] == 4:
        alpha_channel = cover_resized[:, :, 3] / 255.0
        for c in range(3):
            background[y:end_y, x:end_x, c] = (1 - alpha_channel) * background[y:end_y, x:end_x, c] + alpha_channel * cover_resized[:, :, c]
    else:
        background[y:y+h, x:x+w] = cover_resized

def detect_faces(frame, padding=0.2):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    neural_network.setInput(blob)
    detections = neural_network.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Expand box
            padX = int((x2 - x1) * padding)
            padY = int((y2 - y1) * padding)
            x1, y1 = max(0, x1 - padX), max(0, y1 - padY)
            x2, y2 = min(w, x2 + padX), min(h, y2 + padY)

            roi = frame[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(roi, (31, 31), 0)
            frame[y1:y2, x1:x2] = blurred

            if face_covers:
                cover = face_covers[i % len(face_covers)]
                add_face_cover(frame, cover, x1, y1, x2 - x1, y2 - y1)
    return frame

def get_encoder():
    encoders = subprocess.check_output(['ffmpeg', '-hide_banner', '-encoders']).decode()
    return "h264_amf" if "h264_amf" in encoders else "libx264"

def stream_reader(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Could not open stream.")
        return None, None, None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def reader_loop():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if not frame_queue.full():
                frame_queue.put(frame)
        cap.release()

    thread = threading.Thread(target=reader_loop, daemon=True)
    thread.start()
    return fps, width, height

rtmp_input_url = "rtmp://15.156.160.96/live/pi_0001"
rtmp_output_url = "rtmp://15.156.160.96/play/pi_0001"

while True:
    print("Waiting for RTMP stream...")
    fps, width, height = stream_reader(rtmp_input_url)
    if fps is None:
        time.sleep(2)
        continue

    encoder = get_encoder()
    print(f"Using encoder: {encoder}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-"
    ]

    if encoder == "h264_amf":
        ffmpeg_cmd += [
            "-c:v", "h264_amf",
            "-quality", "speed",           # Use AMD-specific quality setting
            "-tune", "zerolatency",
            "-bf", "0",
        ]
    else:
        ffmpeg_cmd += [
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-bf", "0",
            "-threads", "auto",
        ]

    ffmpeg_cmd += [
        "-pix_fmt", "yuv420p",
        "-flush_packets", "1",
        "-rtmp_live", "live",
        "-f", "flv",
        rtmp_output_url
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, bufsize=10**8)
    print("Started FFmpeg, streaming to:", rtmp_output_url)

    try:
        frame_interval = 1.0 / fps
        frame_counter = 0
        last_processed_frame = None

        while True:
            start_time = time.time()
            if frame_queue.empty():
                time.sleep(0.005)
                continue

            frame = frame_queue.get()
            frame_counter += 1

            if frame_counter % 1 == 0:  # Skip every other frame
                last_processed_frame = detect_faces(frame)
            elif last_processed_frame is not None:
                # reuse last processed frame
                pass
            else:
                last_processed_frame = frame

            try:
                process.stdin.write(last_processed_frame.tobytes())
            except BrokenPipeError:
                print("FFmpeg pipe closed.")
                break

            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
    finally:
        process.stdin.close()
        process.wait()
        process.kill()
        print("Restarting stream...")
