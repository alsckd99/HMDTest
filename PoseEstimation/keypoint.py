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

try:
    import onnxruntime  # noqa: F401
    has_onnx = True
except ModuleNotFoundError:
    has_onnx = False
    
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def basic_sharpening(image,strength):
    b = (1 - strength) / 8
    kernel = np.array([[b, b, b],
                                  [b, strength, b],
                                  [b, b, b]])
    
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def compute_dtw_distance(sequence1, sequence2):
    distance, path = fastdtw(sequence1, sequence2, dist=euclidean)
    distance = distance / len(path)
    min_distance = 0
    max_distance = 180
    normalized_distance = (distance - min_distance) / (max_distance - min_distance)
    normalized_distance = np.clip(normalized_distance, 0, 1)
    similarity = 1 - normalized_distance
    accuracy_percentage = similarity * 100
    accuracy_percentage = np.clip(accuracy_percentage, 0, 100)
    return accuracy_percentage, path



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=False,default= "1")
    #parser.add_argument('--input', type=str, required=False,default= "C:\\Users\\user\\Desktop\\Video\\스트레칭_로고제거\\스트레칭_로고제거\\스트레칭_2-4_hamstring curls_231228.mp4")
    #"C:\\Users\\user\\Desktop\\Video\\숄더_로고제거\\숄더_로고제거\\숄더_3-7_trapezius strengthening_231228.mp4"
    parser.add_argument('--output-path', type=str, default="C:\\Users\\user\\Desktop\\Pose_Estimation_Model\\vitpose\\output")
    parser.add_argument('--model', type=str, required=False,default="C:\\Users\\user\\Desktop\\Pose_Estimation_Model\\vitpose\\vitpose-b-coco.pth")
    parser.add_argument('--yolo', type=str, required=False, default="C:\\Users\\user\\Desktop\\Pose_Estimation_Model\\vitpose\\yolov8n.pt")
    parser.add_argument('--dataset', type=str, required=False, default=None)
    parser.add_argument('--det-class', type=str, required=False, default=None)
    parser.add_argument('--model-name', type=str, required=False,default='b', choices=['s', 'b', 'l', 'h'])
    parser.add_argument('--yolo-size', type=int, required=False, default=320)
    parser.add_argument('--conf-threshold', type=float, required=False, default=0.70)
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270],
                        required=False, default=0)
    parser.add_argument('--yolo-step', type=int,
                        required=False, default=1)
    parser.add_argument('--single-pose', default=True)
    parser.add_argument('--show', default=True)
    parser.add_argument('--show-yolo', default=False)
    parser.add_argument('--show-raw-yolo', default=False)
    parser.add_argument('--save-img', default=False)
    parser.add_argument('--save-json', default=False)
    args = parser.parse_args()

    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    # Load Yolo
    yolo = args.yolo
    if yolo is None:
        yolo = 'easy_ViTPose/' + ('yolov8s' + ('.onnx' if has_onnx and not (use_mps or use_cuda) else '.pt'))
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
        cap = cv2.VideoCapture(input_path,cv2.CAP_DSHOW)
    except ValueError:
        assert os.path.isfile(input_path), 'The input file does not exist'
        is_video = input_path[input_path.rfind('.') + 1:].lower() in ['mp4', 'mov']
        cap = cv2.VideoCapture(input_path)
    
    total_frames = 1
    if is_video:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize model
    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=is_video,
                         single_pose=args.single_pose,
                         yolo_step=args.yolo_step)  # type: ignore
    print(f">>> Model loaded: {args.model}")

    print(f'>>> Running inference on {input_path}')
    
    coco_name = ['nose', 'left_eye', 'right_eye', 'left_ear', 
            'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 
            'right_hip', 'left_knee', 'right_knee', 'left_ankle','right_ankle','middle_hip']
    mpii_name = ['head','neck','right_shoulder','right_elbow','right_wrist','left_shoulder','left_elbow','left_wrist',
                 'right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle','chest']
    previous_keypoints = {name: [0.0, 0.0, 0.0] for name in coco_name}
    keypoints = []
    
    while True:
        _, frame = cap.read()
        t0 = time.time()
        keypoints_2D={
            'objects':[]
        }
        try:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_keypoints = model.inference(frame)

            frame_keypoints=np.array(frame_keypoints)
            frame_keypoints[...,[0,1]]=frame_keypoints[...,[1,0]]
            for keypoints in frame_keypoints.tolist():
                obj_dict = {}
                for value,keypoint in zip(coco_name,keypoints):
                    
                    if keypoint[2]<0.5:
                        obj_dict[value] = previous_keypoints[value]
                    else:
                        keypoint[0]=(-keypoint[0])+width
                        keypoint[1]=(-keypoint[1])+height
                        obj_dict[value]=keypoint
                        previous_keypoints[value]=keypoint
                obj_dict['middle_hip'] = ((np.array(obj_dict['right_hip'])+np.array(obj_dict['left_hip']))*0.5).tolist()
                obj_dict.pop('left_ear',None)
                obj_dict.pop('right_ear',None)
                keypoints_2D['objects'].append(obj_dict)
            print("success")
            print(keypoints_2D)
            keypoints.append(frame_keypoints[-1,:,:2])
            image = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]
            cv2.imshow('test',image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("q 키가 눌러져 종료합니다.")
                break
        except Exception as e:
            print(e)
            continue        
    with open("output\\keypoint",'w') as f:
            json.dump(keypoints_2D,f)
    cap.release()
    cv2.destroyAllWindows()
