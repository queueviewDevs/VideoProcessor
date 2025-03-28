import cv2
import numpy as np
import subprocess

neural_network = cv2.dnn.readNetFromCaffe("./deploy.prototxt", "./res10_300x300_ssd_iter_140000.caffemodel") #needed to download pretrained model for face detection from github

def detect_faces(input_frame):
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
            face_roi = input_frame[startY:endY, startX:endX]
            blurred_face = cv2.blur(face_roi, (50, 50))
            input_frame[startY:endY, startX:endX] = blurred_face
            
    return input_frame

rtmp_input_url = "rtmp://queueview.ca/rawfeed/pi_0001"  # Where the Raspberry Pi streams unprocessed video
rtmp_output_url = "rtmp://queueview.ca/processed/pi_0001"   # Where the processed video will be published

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