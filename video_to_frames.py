import cv2
import os

def video_to_frames(video_path, output_folder):
    """
    Divides a video into frames and saves them as images.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    
    while True:
        # Read a frame from the video
        success, image = vidcap.read()
        
        # If the frame was not read successfully, break the loop
        if not success:
            break
        
        # Define the output path for the image
        output_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        
        # Save the frame as an image
        cv2.imwrite(output_path, image)
        print(f"Frame {count} saved to {output_path}")
        
        count += 1
        
    vidcap.release()
    print("Finished extracting frames.")

# --- Script Usage ---
# Path to your video file.
# Make sure to replace this with the correct path to your video.
video_file = "C:/Users/LENOVO/OneDrive/Desktop/AV-RESEARCH/vehicle-project/Untitled video - Made with Clipchamp (7).mp4" 

# Folder where the extracted frames will be saved
output_dir = "frames_output"

# Run the function
video_to_frames(video_file, output_dir)