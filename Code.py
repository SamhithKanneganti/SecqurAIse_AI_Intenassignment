pip install opencv-python opencv-python-headless numpy
pip install opencv-python-headless numpy
import cv2
import numpy as np
import time

# Define the quadrants
def get_quadrant(x, y, frame_width, frame_height):
    if x < frame_width / 2 and y < frame_height / 2:
        return 1
    elif x >= frame_width / 2 and y < frame_height / 2:
        return 2
    elif x < frame_width / 2 and y >= frame_height / 2:
        return 3
    else:
        return 4

# Define color ranges for detection (in HSV space)
color_ranges = {
    "red": [(0, 120, 70), (10, 255, 255)],
    "blue": [(110, 50, 50), (130, 255, 255)],
    "green": [(36, 50, 50), (86, 255, 255)],
}

def track_and_log_events(video_path, output_video_path, log_file_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        print("Error: FPS value is zero.")
        return

    duration = total_frames / fps

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    log_file = open(log_file_path, "w")

    current_quadrant = {}
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # Adjust this threshold based on ball size
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w // 2, y + h // 2
                    quadrant = get_quadrant(cx, cy, frame_width, frame_height)

                    if color not in current_quadrant:
                        current_quadrant[color] = quadrant
                        event_type = "Entry"
                    else:
                        if current_quadrant[color] != quadrant:
                            event_type = "Exit" if current_quadrant[color] < quadrant else "Entry"
                            current_quadrant[color] = quadrant
                        else:
                            continue

                    timestamp = time.time() - start_time
                    log_file.write(f"{timestamp:.2f}, Quadrant {quadrant}, {color.capitalize()}, {event_type}\n")
                    cv2.putText(frame, f"{event_type} {timestamp:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    log_file.close()



# Example usage
track_and_log_events('/content/input_video.mp4', '/content/output_video.avi', '/content/event_log.txt')
