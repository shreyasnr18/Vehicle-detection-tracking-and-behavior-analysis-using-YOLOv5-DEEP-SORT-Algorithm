import cv2

# Make sure this path is exactly the same as in your track_vehicles.py script
video_path = "yolov5/driving_video.mp4"

# Try to open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file. Check file path or video codecs.")
else:
    print("Success: Video file opened. The problem is not the path.")

# Release the video capture object
cap.release()