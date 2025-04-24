import cv2
import mediapipe as mp
import os
import subprocess

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load face covers
face_covers = [cv2.imread(os.path.join("./smileys/", f), cv2.IMREAD_UNCHANGED) for f in sorted(os.listdir("./smileys/")) if f.endswith(".png")]

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

def detect_faces(input_frame):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Convert the frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = input_frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Expand the bounding box
                padding = 0.5  # Adjust padding as needed
                pad_x = int(w * padding)
                pad_y = int(h * padding)
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(iw - x, w + 2 * pad_x)
                h = min(ih - y, h + 2 * pad_y)

                # Blur the detected face
                face_region = input_frame[y:y+h, x:x+w]
                blurred_face = cv2.blur(face_region, (50, 50))
                input_frame[y:y+h, x:x+w] = blurred_face

                # Add the face cover
                cover = face_covers[0]  # Use the first face cover (or cycle through them if needed)
                add_face_cover(input_frame, cover, x, y, w, h)

        return input_frame

rtmp_input_url = "rtmp://localhost/live/pi_0001"  # Where the Raspberry Pi streams unprocessed video
rtmp_output_url = "rtmp://localhost/play/pi_0001"   # Where the processed video will be published

while True:
    while True:
        print("Attempting to look for video")
        stream = cv2.VideoCapture(rtmp_input_url)

        if not stream.isOpened():
            print("No Stream Detected. Check camera connection.")
            continue
        else:
            break

    # Retrieve stream properties for proper encoding
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = stream.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # default FPS if detection fails

    # Setup FFmpeg process to push the processed frames to the RTMP output
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",               # Read video from stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-f", "flv",
        rtmp_output_url          # Push the processed stream to this RTMP URL
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    print("Streaming processed video to:", rtmp_output_url)

    while True:
        ret, frame = stream.read()
        if not ret:
            print("stream.read() returned false.")
            print("stream has ended or camera error.")
            break

        processed_frame = detect_faces(frame)

        try:
            process.stdin.write(processed_frame.tobytes())
        except BrokenPipeError:
            print("FFmpeg process closed the pipe.")
            break


    stream.release()
    process.stdin.close()
    process.wait()
    process.kill()