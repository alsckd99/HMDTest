import argparse
import json
import os
import time
import base64
from PIL import Image
import cv2
import numpy as np
import torch
import tqdm
import socket
from PIL import Image
from torchvision import transforms
from vitpose.vit_utils.inference import NumpyEncoder, VideoReader
from vitpose.inference import VitInference
from vitpose.vit_utils.visualization import joints_dict
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math

try:
    import onnxruntime  # noqa: F401
    has_onnx = True
except ModuleNotFoundError:
    has_onnx = False

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input', type=str, required=False,default= "1")
    parser.add_argument('--input', type=str, required=False,default= os.path.join(script_dir, "Video\\자가보조\\1.어깨굴곡-신전_자가보조관절.mp4"))
    parser.add_argument('--output-path', type=str, default=os.path.join(script_dir, "output"))
    parser.add_argument('--model', type=str, required=False,default=os.path.join(script_dir, "ckpts\\vitpose\\vitpose-b-coco.pth"))
    parser.add_argument('--yolo', type=str, required=False, default=os.path.join(script_dir,"ckpts\\yolo\\yolo11m.pt"))
    parser.add_argument('--dataset', type=str, required=False, default=None)
    parser.add_argument('--det-class', type=str, required=False, default=None)
    parser.add_argument('--model-name', type=str, required=False,default='b', choices=['s', 'b', 'l', 'h'])
    parser.add_argument('--yolo-size', type=int, required=False, default=320)
    parser.add_argument('--conf-threshold', type=float, required=False, default=0.50)
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270],
                        required=False, default=0)
    parser.add_argument('--yolo-step', type=int,
                        required=False, default=1)
    parser.add_argument('--single-pose', default=True)
    parser.add_argument('--show', default=False)
    parser.add_argument('--show-yolo', default=False)
    parser.add_argument('--show-raw-yolo', default=False)
    parser.add_argument('--save-img', default=False)
    parser.add_argument('--save-json', default=False)
    args = parser.parse_args()

    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    input_path = args.input

    # Load Yolo
    yolo = args.yolo
    if yolo is None:
        yolo = os.path.join(script_dir, "ckpts", "yolo", "yolo11m.pt")
        
    # Since we are processing video files, is_video should be True
    is_video = True
        
    # Initialize model
    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=is_video,
                         single_pose=args.single_pose,
                         yolo_step=args.yolo_step)  # type: ignore
    print(f">>> Model loaded: {args.model}")

    if os.path.isdir(args.input):
        video_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        input_paths = [os.path.join(args.input, f) for f in video_files]
    else:
        input_paths = [args.input]

    for video_path in input_paths:
        print(f'>>> Running inference on {video_path}')
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue

        if args.output_path:
            output_dir = args.output_path
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(output_dir, f"{base_name}_result.mp4")

        coco_name = ['nose', 'left_eye', 'right_eye', 'left_ear', 
                'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
                'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 
            'right_hip', 'left_knee', 'right_knee', 'left_ankle','right_ankle','middle_hip']
    mpii_name = ["right_ankle","right_knee","right_hip",
                "left_hip","left_knee","left_ankle",
                "pelvis","thorax","neck","head",
                "right_wrist","right_elbow","right_shoulder",
                "left_shoulder","left_elbow","left_wrist"]
    previous_keypoints = {name: [0.0, 0.0, 0.0] for name in mpii_name}
    keypoints = []
    fps = []
    angle1_list=[]
    angle2_list=[]

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t0 = time.time()
            keypoints_2D={
                'objects':[]
            }
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_keypoints = model.inference(frame)
                frame_keypoints=np.array(frame_keypoints)
                frame_keypoints[...,[0,1]]=frame_keypoints[...,[1,0]]
                image = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]
                speed = 1/(time.time()-t0)
                print(speed)
                if args.show:
                    cv2.imshow('preview', image)
                    cv2.waitKey(1)
            except Exception as e:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if args.show:
                    cv2.imshow('preview', frame)
                    cv2.waitKey(1)
                print(e)
                continue        
            else:
                cap.release()
    cv2.destroyAllWindows()
