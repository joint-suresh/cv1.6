import cv2
import os

video_path = 'video.mp4'
output_dir = 'video_frames'
os.makedirs(output_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(output_dir, f'frame_{count:04d}.jpg'), frame)
    count += 1
cap.release()