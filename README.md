# USBFaceRec
Simple USB camera facial recognition for the RPI. This is a modified version to optimize FPS preformance when using a USB camera instead of a picamera.

These scripts have been modified to only support USB cameras as a video source. It will not support the camera header on the RPI as I removed picamera2 from it.

I followed a tutorial from Core Electronics on YouTube for initial setup of the python virtual enviroment and installing the system-site-package: face_rec
Follow the turorial on setup, then when the time comes to run the python recognition scrips for training or gerneral use you can simply use my modded version.

Note: You can download the Face Recognition ZIP file that contains the facial recognition scripts. You can find the original ZIP from Core Electronics Here: 
Core Electronics full orginal guide: https://core-electronics.com.au/guides/raspberry-pi/face-recognition-with-raspberry-pi-and-opencv/

Then YouTube tutorial: Core Electronics tutorial: https://www.youtube.com/watch?v=3TUlJrRJUeM
Follow Core Electronics tutorial as it was very detailed and easy to understand for anyone getting into python and models.

You can either clone this repository with git or download directly from GitHub then extract the folder from the ZIP and then you are good to go. 
