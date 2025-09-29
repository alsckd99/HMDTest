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
    parser.add_argument('--conf-threshold', type=float, required=False, default=0.75)
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270],
                        required=False, default=0)
    parser.add_argument('--yolo-step', type=int,
                        required=False, default=1)
    parser.add_argument('--single-pose', default=True)
    parser.add_argument('--show', default=True)
    parser.add_argument('--show-yolo', default=True)
    parser.add_argument('--show-raw-yolo', default=True)
    parser.add_argument('--save-img', default=False)
    parser.add_argument('--save-json', default=False)
    args = parser.parse_args()

    use_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    keypoints_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    keypoints_Port = ("127.0.0.1",5252)
    
    image_sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    image_Port=("127.0.0.1",5253)
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
    
    seriesA = np.array([0, 1, 2, 3]).reshape(-1, 1)
    seriesB = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    a,b = compute_dtw_distance(seriesA,seriesB)
    wait = 0
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
    fps = []
    tot_time = 0.
    angle1_list=[]
    angle2_list=[]
    
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')         # 인코딩 포맷
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # size = (int(width), int (height))                   # 프레임 크기
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # fps = float(fps) if fps > 0 else 20.0 
    # out = cv2.VideoWriter("C:\\Users\\user\\Desktop\\Pose_Estimation_Model\\vitpose\\output\\result.mp4", fourcc, fps, size) 
    
    while True:
        _, frame = cap.read()
        t0 = time.time()
        keypoints_2D={
            'objects':[]
        }
        try:
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = basic_sharpening(frame,12)
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
            json_data = json.dumps(keypoints_2D)
            keypoints_sock.sendto(json_data.encode('utf-8'), keypoints_Port)
            image = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]
            #encoded, buffer = cv2.imencode('.jpg', image)
            #data = buffer.tobytes()
            #image_sock.sendto(data,image_Port)
            print("success")
            keypoints.append(frame_keypoints[-1,:,:2])
            delta = time.time() - t0
            tot_time += delta
            fps.append(delta)
            if args.show:
                cv2.imshow('preview', image)
                cv2.waitKey(1)
                
            
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_keypoints = model.inference(frame)

            # image = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]
            
            # out.write(image)
            # cv2.imshow('test',image)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     print("q 키가 눌러져 종료합니다.")
            #     break
            
            # ------------------------------------------------------------------------------------
            # for keypoints in frame_keypoints.tolist():
            #     angle1 = calculate_angle(keypoints[11][:2],keypoints[5][:2],keypoints[9][:2])
            #     angle2 = calculate_angle(keypoints[13][:2],keypoints[11][:2],keypoints[5][:2])
            #     angle1_list.append(angle1)
            #     angle2_list.append(angle2)

            # Draw the poses and save the output img
            # if args.show or args.save_img:
            #     # Draw result and transform to BGR
            #     img = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]

            #     if args.save_img:
            #         # TODO: If exists add (1), (2), ...
            #         if is_video:
            #             out_writer.write(img)
            #         else:
            #             print('>>> Saving output image')
            #             cv2.imwrite(output_path_img, img)


         
        except Exception as e:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #encoded, buffer = cv2.imencode('.jpg', frame)
            #data = buffer.tobytes()
            #image_sock.sendto(data,image_Port)
            if args.show:
                cv2.imshow('preview', frame)
                cv2.waitKey(1)
            print(e)
            continue
            
        
    # result_list = [angle1_list,angle2_list]
    # print(result_list[0])
    # if is_video:
    #     tot_poses = sum(len(k) for k in keypoints)
    #     print(f'>>> Mean inference FPS: {1 / np.mean(fps):.2f}')
    #     print(f'>>> Total poses predicted: {tot_poses} mean per frame: '
    #           f'{(tot_poses / (ith + 1)):.2f}')
    #     print(f'>>> Mean FPS per pose: {(tot_poses / tot_time):.2f}')

    # if args.save_json:
    #     print('>>> Saving output json')
    #     with open('Trapezius_Strengthening', 'w') as f:
    #         json.dump(result_list, f)
    #     #keypoints = np.array(keypoints)
        
    #     #np.savez('output.npz',keypoints, allow_pickle=True)

    # if is_video and args.save_img:
    #     out.release()
    # cv2.destroyAllWindows()
    # out.release()

    cap.release()
    cv2.destroyAllWindows()
