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

stream = cv2.VideoCapture(rtmp_input_url)

if not stream.isOpened():
    print("No Stream Detected. Check camera connection.")
    exit()

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

while(True):
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
    # cv2.imshow("webcam", frame)

    # if cv2.waitKey(1) == ord('q'):
    #     break

stream.release()
process.stdin.close()
process.wait()
# cv2.destroyAllWindows()


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #get face model weights from cv2
# profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
# def detect_faces(input_frame):
#     greyscale_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY) #convert frame to greyscale for easier processing
#     faces = face_cascade.detectMultiScale(greyscale_frame, 1.3, 8) #detect faces using cv2 function -- 1.3 = zoom in on frame, 5 = strength/strictness/quality of photo for face recognition
#     profiles = profile_cascade.detectMultiScale(greyscale_frame, 1.3, 8)

#     for(x, y, w, h) in faces: 
#         face_region = input_frame[y:y+h, x:x+w]
#         blurred_face = cv2.blur(face_region, (50, 50))
#         input_frame[y:y+h, x:x+w] = blurred_face

#     for(x, y, w, h) in profiles: 
#         face_region = input_frame[y:y+h, x:x+w]
#         blurred_face = cv2.blur(face_region, (50, 50))
#         input_frame[y:y+h, x:x+w] = blurred_face

#     return input_frame






#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# BELOW CODE WORKS FOR FACES THAT LOOK DIRECTLY AT CAMERA
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# import cv2


# # face_cover_pic = cv2.imread('face_cover.jpg', cv2.IMREAD_UNCHANGED) #import image to cover face

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #get face model weights from cv2
# def detect_faces(input_frame):
#     greyscale_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY) #convert frame to greyscale for easier processing
#     faces = face_cascade.detectMultiScale(greyscale_frame, 1.3, 8) #detect faces using cv2 function -- 1.3 = zoom in on frame, 5 = strength/strictness/quality of photo for face recognition

#     for(x, y, w, h) in faces: 
#         face_region = input_frame[y:y+h, x:x+w]
#         blurred_face = cv2.blur(face_region, (50, 50))
#         input_frame[y:y+h, x:x+w] = blurred_face

        
#         #-------------------------------------------------------------------------------------------
#         #---------------------------- cover face with emoji ----------------------------------------
#         #face_cover = cv2.resize(face_cover_pic, (w, h)) #resize face cover image to be w x h of detected face
#         #input_frame[y:y+h, x:x+w] = face_cover # replace the detected face portion with the face cover image

#         #-------------------------------------------------------------------------------------------
#         #---------------------------- add border around face ---------------------------------------
#         #input_frame = cv2.rectangle(input_frame, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=-1) #add border around face

#     return input_frame

# stream = cv2.VideoCapture(0)

# if not stream.isOpened():
#     print("No Stream Detected. Check camera connection.")
#     exit()

# while(True):
#     ret, frame = stream.read()
#     if not ret:
#         print("stream.read() returned false.")
#         print("stream has ended or camera error.")
#         break

#     frame = detect_faces(frame)
#     cv2.imshow("webcam", frame)

#     if cv2.waitKey(1) == ord('q'):
#         break

# stream.release()
# cv2.destroyAllWindows()


