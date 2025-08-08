import cv2
import numpy as np
from collections import OrderedDict

# CentroidTracker class to keep track of objects and their IDs across frames
class CentroidTracker:
    def __init__(self, max_disappeared=50):
        self.next_object_id = 0
        self.objects = OrderedDict()  # Object ID and their centroids
        self.disappeared = OrderedDict()  # Tracks how long an object has been missing
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        # Register a new object
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        # Deregister an object
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)

        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distance between current centroids and input centroids
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


# Load the pre-trained MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Labels for the classes in the COCO dataset, index 15 is 'person'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Load the video stream
cap = cv2.VideoCapture("test.mp4")

# Counters for people moving up and down
up_count = 0
down_count = 0

# Create an instance of CentroidTracker
ct = CentroidTracker()

# Dictionary to store the previous x-coordinate for each object
previous_x = {}

# Set horizontal line position (center of the screen)
line_position = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Prepare the input blob for object detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Perform forward pass to get detections
    detections = net.forward()

    # Initialize a list to hold detected centroids
    centroids = []

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Only process confident detections (greater than 50%)
        if confidence > 0.5:
            # Get the class label
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] != "person":
                continue

            # Compute the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Calculate the centroid
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            centroids.append(centroid)

            # Draw the bounding box and label
            label = f"Person {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Update the centroid tracker with the new centroids
    objects = ct.update(centroids)

    # Loop over the tracked objects
    for (object_id, centroid) in objects.items():
        current_x = centroid[0]

        # Check if this object has been seen before
        if object_id in previous_x:
            prev_x = previous_x[object_id]

            # Check if the person is moving up (from below the line to above it)
            if prev_x > line_position and current_x < line_position:
                up_count += 1
                print(f"Person {object_id} moved up.")

            # Check if the person is moving down (from above the line to below it)
            elif prev_x < line_position and current_x > line_position:
                down_count += 1
                print(f"Person {object_id} moved down.")

        # Store the current x-coordinate as the previous x-coordinate for the next frame
        previous_x[object_id] = current_x

    # Draw the horizontal line across the frame
    cv2.line(frame, (line_position, 0), (line_position, h), (0, 0, 255), 2)

    # Display the up and down counts on the frame
    cv2.putText(frame, f"Up: {up_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Down: {down_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame with detections and counters
    cv2.imshow("People Counting", frame)

    # Break the loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
