import argparse
import json
import os
import time
import cv2
import numpy as np
import torch
import socket
from vitpose.inference_tracker import VitInference
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import math
from ultralytics import YOLO

try:
    import onnxruntime  # noqa: F401
    has_onnx = True
except ModuleNotFoundError:
    has_onnx = False
    

if __name__ == "__main__":
    print(np.__version__)
    print(cv2.__version__)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False,default= "1")
    parser.add_argument('--output-path', type=str, default=os.path.join(script_dir, "output"))
    parser.add_argument('--model', type=str, required=False,default=os.path.join(script_dir, "ckpts\\vitpose\\vitpose-b-coco.pth"))
    parser.add_argument('--yolo', type=str, required=False, default=os.path.join(script_dir,"ckpts\\yolo\\yolo11x.pt"))
    parser.add_argument('--dataset', type=str, required=False, default=None)
    parser.add_argument('--det-class', type=str, required=False, default=None)
    parser.add_argument('--model-name', type=str, required=False, default='b', 
                        choices=['s', 'b', 'l', 'h'])
    parser.add_argument('--yolo-size', type=int, required=False, default=320)
    parser.add_argument('--conf-threshold', type=float, required=False, default=0.65)
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270],
                        required=False, default=0)
    parser.add_argument('--show', default=True)
    parser.add_argument('--show-yolo', default=True)
    parser.add_argument('--show-raw-yolo', default=False)
    parser.add_argument('--save-img', default=False)
    parser.add_argument('--save-json', default=False)
    args = parser.parse_args()

    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    
    keypoints_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    keypoints_Port = ("127.0.0.1", 5252)
    image_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    image_Port = ("127.0.0.1", 5253)    
    
    # Load Yolo
    yolo = args.yolo
    input_path = args.input
    ext = input_path[input_path.rfind('.'):]
    assert not (args.save_img or args.save_json) or args.output_path, \
        'Specify an output path if using save-img or save-json flags'
    output_path = args.output_path
    
    if output_path:
        if os.path.isdir(output_path):
            save_name_img = os.path.basename(input_path).replace(ext, f"_result{ext}")
            save_name_json = os.path.basename(input_path).replace(ext, "_result.json")
            output_path_img = os.path.join(output_path, save_name_img)
            output_path_json = os.path.join(output_path, save_name_json)
        else:
            output_path_img = output_path + f'{ext}'
            output_path_json = output_path + '.json'

    try:  # Check if is webcam
        input_path = int(input_path)
        is_video = True
        cap = cv2.VideoCapture(input_path, cv2.CAP_DSHOW)
    except ValueError:
        assert os.path.isfile(input_path), 'The input file does not exist'
        is_video = input_path[input_path.rfind('.') + 1:].lower() in ['mp4', 'mov']
        cap = cv2.VideoCapture(input_path)
    wait = 0
    total_frames = 1
    if is_video:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize model
    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=is_video,)
    print(f">>> Model loaded: {args.model}")
    print(f'>>> Running inference on {input_path}')
    
    coco_name = ['nose', 'left_eye', 'right_eye', 'left_ear', 
                 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
                 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 
                 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'middle_hip']
    mpii_name = ["right_ankle", "right_knee", "right_hip",
                 "left_hip", "left_knee", "left_ankle",
                 "pelvis", "thorax", "neck", "head",
                 "right_wrist", "right_elbow", "right_shoulder",
                 "left_shoulder", "left_elbow", "left_wrist"]
    
    previous_keypoints = {name: [0.0, 0.0, 0.0] for name in coco_name}
    keypoints = []
    fps = []
    angle1_list = []
    angle2_list = []
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')         # 인코딩 포맷
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # resize_width = width/2
    # resize_height = height/2
    # size = (int(resize_width), int (resize_height))                   
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps > 0 else 20.0 
    
    # out = cv2.VideoWriter(os.path.join(script_dir,"output\\result.mp4"), fourcc, fps, size) 
     
    yolo = YOLO(yolo)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            t0 = time.time()
            keypoints_2D = {
                'objects': []
            }
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # frame = cv2.resize(frame, (int(resize_width), int(resize_height)))
                frame_keypoints,anotated_frame = model.inference(frame)
                frame_keypoints = np.array(frame_keypoints)
                frame_keypoints[..., [0, 1]] = frame_keypoints[..., [1, 0]]
                for idx, keypoints in enumerate(frame_keypoints.tolist()):
                    obj_dict = {}
                    # ID 정보 추가
                    obj_dict['id'] = idx  # 순차적인 ID 사용
                    
                    # 해당 ID의 이전 키포인트가 없으면 초기화
                    if idx not in previous_keypoints:
                        previous_keypoints[idx] = {name: [0.0, 0.0, 0.0] 
                                                  for name in coco_name}
                    
                    for value, keypoint in zip(coco_name, keypoints):
                        
                        if keypoint[2] < 0.5:
                            obj_dict[value] = previous_keypoints[idx][value]
                        else:
                            keypoint[0] = (-keypoint[0]) + width
                            keypoint[1] = (-keypoint[1]) + height
                            obj_dict[value] = keypoint
                            previous_keypoints[idx][value] = keypoint
                            
                    keypoints_2D['objects'].append(obj_dict)
                #print(keypoints_2D)
                json_data = json.dumps(keypoints_2D)
                keypoints_sock.sendto(json_data.encode('utf-8'), keypoints_Port)
                image = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]
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
