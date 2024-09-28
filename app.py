import cv2

# Start capturing video from the webcam
# (if you have multiple cameras, try changing the index to 1, 2, etc.)
cap = cv2.VideoCapture(0)

while True:
    # Capture each frame from the webcam
    ret, frame = cap.read()
    
    # Check if the frame was read correctly
    if not ret:
        print("Error: Missing frame")
        continue  # Skip to the next iteration

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray_frame, 100, 200)

    # Show the original frame and the edges
    cv2.imshow('Webcam - Original', frame)
    cv2.imshow('Webcam - Edges', edges)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
