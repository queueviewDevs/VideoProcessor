import cv2
import numpy as np
import os
import subprocess

neural_network = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "./res10_300x300_ssd_iter_140000.caffemodel") #needed to download pretrained model for face detection from github
face_covers = [cv2.imread(os.path.join("./smileys/", f), cv2.IMREAD_UNCHANGED) for f in sorted(os.listdir("./smileys/")) if f.endswith(".png")]

def add_face_cover(background, cover, x, y, w, h, scale=1.5):
    new_w = int(w * scale)
    new_h = int(h * scale)
    cover_resized = cv2.resize(cover, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x -= (new_w - w) // 2
    y -= (new_h - h) // 2

    # Ensure coordinates stay within frame boundaries
    x, y = max(0, x), max(0, y)
    end_x, end_y = min(background.shape[1], x + new_w), min(background.shape[0], y + new_h)

    # Resize again if it overflows the frame
    cover_resized = cover_resized[:end_y - y, :end_x - x]
    if cover_resized.shape[2] == 4:  # Check if overlay has an alpha channel
        alpha_channel = cover_resized[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]
        for c in range(3):  # Blend each color channel
            background[y:end_y, x:end_x, c] = (1 - alpha_channel) * background[y:end_y, x:end_x, c] + alpha_channel * cover_resized[:, :, c]
    else:
        background[y:y+h, x:x+w] = cover_resized  # If no alpha, direct replacement

def detect_faces(input_frame, padding=0.2):
    (h, w) = input_frame.shape[:2] #extract height and width of frame -- .shape has parameters (h, w, colour_channels_used)
    blob = cv2.dnn.blobFromImage(input_frame, 1.0, (300, 300), (104.0, 177.0, 123.0)) #preprocess frame -- (input_frame, scale_facotr, image_resize_to, mean_subtraction_values_for_rgb)
    neural_network.setInput(blob) #set blob (preprocessed frame) as input for the neural network
    faces_detected = neural_network.forward() #perform forward pass through the network to detect all faces

    for i in range(faces_detected.shape[2]): #iterate through all detected faces .shape[2] contains num detected
        confidence = faces_detected[0, 0, i, 2] #faces_detected[batch_id, object_class, detection_index, attribute_index] -- attribute_index = 2 stores confidence level of detection
        if confidence > 0.3: #set threshold for confidence
            detected_area = faces_detected[0, 0, i, 3:7] * np.array([w, h, w, h]) #scale image back to original size -- faces_detected[batch_id, object_class, detection_index, attribute_index] -- attribute_index = 3:7 stores bounding box coordinates of detected face
            (startX, startY, endX, endY) = detected_area.astype("int") #convert coordinates to integers
            startX, startY = max(0, startX), max(0, startY) #ensure coordinates are within the frame bounds
            endX, endY = min(w, endX), min(h, endY) #ensure coordinates are within the frame bounds
            # -------- Expand bounding box --------
            padX = int((endX - startX) * padding)
            padY = int((endY - startY) * padding)

            startX = max(0, startX - padX)
            startY = max(0, startY - padY)
            endX = min(w, endX + padX)
            endY = min(h, endY + padY)
            # -------------------------------------
            face_roi = input_frame[startY:endY, startX:endX]
            blurred_face = cv2.blur(face_roi, (50, 50))
            input_frame[startY:endY, startX:endX] = blurred_face
            #----------- add face cover ----------------
            cover = face_covers[i % len(face_covers)] #cycle through face covers
            add_face_cover(input_frame, cover, startX, startY, endX - startX, endY - startY)
            #-------------------------------------------
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