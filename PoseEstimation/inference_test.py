import argparse
import json
import os
import sys

import cv2
import torch
import numpy as np
import socket
import time

from vitpose.vit_utils.inference import NumpyEncoder
from vitpose.inference_test import VitInference

# --- C#과 통신하기 위한 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STOP_SIGNAL_FILE_PATH = os.path.join(SCRIPT_DIR, 'stop_signal.txt')
START_SIGNAL_FILE_PATH = os.path.join(SCRIPT_DIR, 'start_signal.txt')


def main():
    parser = argparse.ArgumentParser()
    # input_video 관련 인자 제거
    parser.add_argument(
        '--input', type=str,  default="1",
        help='webcam ID for pose estimation')
    parser.add_argument(
        '--output-path', type=str, default='output',
        help='Output path for the results.')
    parser.add_argument(
        '--model', type=str,  default="ckpts/vitpose/vitpose-b-coco.pth",
        help='checkpoint path of the model')
    parser.add_argument(
        '--yolo', type=str, required=False,
        default="ckpts\\yolo\\yolo11m.pt",
        help='checkpoint path of the yolo model')
    
    dataset_help = (
        'Name of the dataset. If None it is extracted from the file name.'
        '["coco", "coco_25", "wholebody", "mpii", '
        '"ap10k", "apt36k", "aic", "custom"]')
    parser.add_argument(
        '--dataset', type=str, required=False, default='coco',
        help=dataset_help)
    
    det_class_help = (
        '["human", "cat", "dog", "horse", "sheep", '
        '"cow", "elephant", "bear", "zebra", "giraffe", "animals"]')
    parser.add_argument(
        '--det-class', type=str, required=False, default='human',
        help=det_class_help)
    parser.add_argument(
        '--model-name', type=str, required=False, default="b",
        choices=['s', 'b', 'l', 'h'],
        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument(
        '--yolo-size', type=int, required=False, default=320,
        help='YOLO image size during inference')
    
    conf_threshold_help = (
        'Minimum confidence for keypoints to be drawn. [0, 1] range')
    parser.add_argument(
        '--conf-threshold', type=float, required=False, default=0.5,
        help=conf_threshold_help)

    rotate_help = (
        'Rotate the image of [90, 180, 270] '
        'degress counterclockwise')
    parser.add_argument(
        '--rotate', type=int, choices=[0, 180, 270],
        required=False, default=0, help=rotate_help)
    parser.add_argument(
        '--single-pose', default=False, action='store_true',
        help='Do not use SORT tracker because single pose is expected')
    parser.add_argument(
        '--show', default=True, action='store_true',
        help='preview result during inference')
    parser.add_argument(
        '--show-yolo', default=True, action='store_true',
        help='draw yolo results')
    
    show_raw_yolo_help = (
        'draw yolo result before that SORT is applied for tracking'
        ' (only valid during video inference)')
    parser.add_argument(
        '--show-raw-yolo', default=False, action='store_true',
        help=show_raw_yolo_help)
    parser.add_argument(
        '--save-img', default=False, action='store_true',
        help='save image results')
    parser.add_argument(
        '--save-json', default=True, action='store_true',
        help='save json results')
    args = parser.parse_args()

    # use_cuda = torch.cuda.is_available()
    keypoints_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    keypoints_Port = ("127.0.0.1", 5252)
    yolo = args.yolo

    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=True)
    print(f">>> Model loaded: {args.model}")

    # --- Setup webcam capture ---
    try:
        webcam_id = int(args.input)
    except ValueError:
        print(f"Error: --input must be a webcam ID (integer). Got {args.input}")
        sys.exit(1)

    cap_webcam = cv2.VideoCapture(webcam_id, cv2.CAP_DSHOW)
    cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap_webcam.isOpened():
        print(f"Error opening webcam: {webcam_id}")
        sys.exit(1)


    webcam_width = cap_webcam.get(cv2.CAP_PROP_FRAME_WIDTH)
    webcam_height = cap_webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f">>> Processing webcam: {webcam_id} ({int(webcam_width)}x{int(webcam_height)})")

    # --- 파일명 자동 증가 함수 ---
    def get_next_filename(base_dir, base_name, extension):
        """파일명을 자동으로 증가시켜 중복을 방지하는 함수"""
        counter = 1
        while True:
            filename = f"{base_name}_{counter}{extension}"
            filepath = os.path.join(base_dir, filename)
            if not os.path.exists(filepath):
                return filepath, filename
            counter += 1

    # --- VideoWriter 설정 ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap_webcam.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps > 0 else 20.0
    # 90도 회전으로 인해 너비와 높이를 교체합니다.
    size = (int(webcam_width), int(webcam_height))

    output_dir = os.path.join(SCRIPT_DIR, "KTPFormer","output","vitpose")
    os.makedirs(output_dir, exist_ok=True)
    
    # 자동 증가 파일명 생성
    output_path, output_filename = get_next_filename(output_dir, "train7", ".mp4")
    json_filename = output_filename.replace(".mp4", ".json")
    output_json_path = os.path.join(output_dir, json_filename)

    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    start_signal_sent = False  # 시작 신호를 한 번만 보내기 위한 플래그
    frame_idx = 0
    fps = 0   
    all_keypoints_for_json = []
    try:
        # --- Main loop ---
        while True:
            if os.path.exists(STOP_SIGNAL_FILE_PATH):
                print(">>> Python: 종료 신호를 감지했습니다.")
                break

            ret_webcam, frame_webcam = cap_webcam.read()
            if not ret_webcam:
                print(">>> Python: 웹캠 프레임을 가져오지 못했습니다. 루프를 종료합니다.")
                break

            if not start_signal_sent:
                print(">>> Python: 녹화를 시작합니다.")
                with open(START_SIGNAL_FILE_PATH, 'w') as f:
                    f.write('start')
                start_signal_sent = True

            out.write(frame_webcam)

            # --- 동작 인식 포인트 추출 부분 주석 처리 ---
            try:
                start_time = time.time()
                frame_webcam_rgb = cv2.cvtColor(frame_webcam, cv2.COLOR_BGR2RGB)
                frame_keypoints = model.inference(frame_webcam_rgb)
                frame_keypoints=np.array(frame_keypoints)
                frame_keypoints[...,[0,1]]=frame_keypoints[...,[1,0]]
                if frame_keypoints is not None and len(frame_keypoints) > 0:
                    person_keypoints = frame_keypoints[0] # Assuming one person
                    frame_data = {
                        "idx": [0, frame_idx],
                        "keypoints": person_keypoints.flatten().tolist()
                    }
                    all_keypoints_for_json.append(frame_data)
            
                if args.show:
                    image_to_show = model.draw(args.show_yolo, args.show_raw_yolo, args.conf_threshold)[..., ::-1]
                    cv2.imshow('Webcam Keypoints', image_to_show)
                frame_idx += 1
                end_time = time.time()
                fps = 1.0 / (end_time - start_time)
            except Exception as e:
                print(f"Webcam processing error: {e}")
                if args.show:
                    cv2.imshow('Webcam Keypoints', frame_webcam)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    finally:
        cap_webcam.release()
        out.release()
        cv2.destroyAllWindows()
        print(f">>> Python: 녹화 영상을 저장했습니다. 경로: {output_path}")

        if args.save_json and all_keypoints_for_json:
            with open(output_json_path, 'w') as f:
                json.dump(all_keypoints_for_json, f, cls=NumpyEncoder)
            print(f"총 {frame_idx} 프레임의 동작 인식 데이터를 JSON 파일로 저장하였습니다. {output_json_path}")

        # 자신이 만든 시작/종료 신호 파일을 모두 정리합니다.
        if os.path.exists(START_SIGNAL_FILE_PATH):
            os.remove(START_SIGNAL_FILE_PATH)
        if os.path.exists(STOP_SIGNAL_FILE_PATH):
            os.remove(STOP_SIGNAL_FILE_PATH)

        print(">>> Python 스크립트가 종료되었습니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()