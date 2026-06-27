import cv2

# This function will be called when you click the mouse
def click_event(event, x, y, flags, params):
    # Check if a left-click event occurred
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinates to the terminal
        print(f"({x}, {y})")
        # Draw a circle on the image where you clicked
        cv2.circle(img, (x, y), 3, (0, 0, 255), -1)
        # Update the image window
        cv2.imshow('Image', img)

# Load your screenshot of the video
# Make sure to place the screenshot file in the same folder as this script
img = cv2.imread('ss_of_driving_video.png')

# Check if the image was loaded correctly
if img is None:
    print("Error: Could not load image.")
    exit()

# Create a window to display the image
cv2.namedWindow('Image')
# Set the mouse callback to your click_event function
cv2.setMouseCallback('Image', click_event)

# Display the image and wait for key presses
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()