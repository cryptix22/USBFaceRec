#This is a modified version of Face Recognition. This modified version contains code from the authors that originally made this program. Licensing can be found in the main Face Recognition folder.
#This version does not support picamera. It is designed to support USB cameras.
#The speed of face_rec will depend on the hardware used. It runs very fast on the RPI5. 

import face_recognition
import cv2
import numpy as np
import time
import pickle

#Load the pre-trained models

print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

#Init the OpenCV USB video capture stream. Usually 0 in the cv2.VideoCapture() is the deafult device, you may need to change this.
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

#Init variables
cv_scaler = 4  # This has to be a whole number
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def process_frame(frame):
    global face_locations, face_encodings, face_names
    #Reduce the frame size to improve preformance  
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    
    #Convert the image from BGR to RGB as face_recognition uses RGB
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    #Detect face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')
    
    face_names = []
    for face_encoding in face_encodings:
        #Compare the face in the frame to the known face models
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        #Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    
    return frame

def draw_results(frame):
    #Draw a box around the detected face and bind to it, as well as set a label
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        #Scale back up face locations since the frame was resized
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        #Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        
        #Draw a name tag/rectangle for current face
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

while True:
    #Capture the frames from the USB camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    #Process the frames
    processed_frame = process_frame(frame)
    
    #Display the detection resuslt within the capture frame
    display_frame = draw_results(processed_frame)
    
    #Calculate the Frames per Second(FPS) of the camera stream
    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", 
                (display_frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    #To wrap it all togeather, display the USB video stream with recognition results
    cv2.imshow('Video', display_frame)
    
    #To exit the face_recognition proccess, press "q"
    #You can also stop it within the Thonny GUI
    if cv2.waitKey(1) == ord("q"):
        break

#Release the OpenCV stream and kill all windows
cap.release()
cv2.destroyAllWindows()
