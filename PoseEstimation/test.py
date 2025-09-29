import argparse
import os
import time
from PIL import Image
import cv2
import numpy as np
import torch
from collections import defaultdict
from ultralytics import YOLO

video_path = "Video/자가보조/1.어깨굴곡-신전_자가보조관절.mp4"
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
    if success:
        model=YOLO("yolov10n.pt")
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLO11 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

while cap.isOpened():

    success, frame = cap.read()
    if success:
        cv2.imshow('Webcam Recording...', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')         # 인코딩 포맷
# width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# size = (int(width), int (height))                   # 프레임 크기
# fps = cap.get(cv2.CAP_PROP_FPS)
# fps = float(fps) if fps > 0 else 20.0 
# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(script_dir, "output")
# os.makedirs(output_dir, exist_ok=True)
# output_path = os.path.join(output_dir, "result.mp4")

# out = cv2.VideoWriter(output_path, fourcc, fps, size)

# print("녹화를 시작합니다. 'q' 키를 누르면 중단됩니다.")

# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         # 프레임 저장 및 화면 출력
#         out.write(frame)
#         cv2.imshow('Webcam Recording...', frame)

#         # 'q' 키를 누르면 루프 종료
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         break

# # 리소스 해제
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# print(f"'{output_path}'에 영상이 저장되었습니다.")
