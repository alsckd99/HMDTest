import argparse
import json
import os
import sys

import cv2
import torch
import numpy as np

from vitpose.vit_utils.inference import NumpyEncoder
from vitpose.inference_test import VitInference

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    default_input = os.path.join(SCRIPT_DIR, "Video/자가보조/1.어깨굴곡-신전_자가보조관절.mp4")
    parser.add_argument(
        '--input', type=str, default=default_input,
        help='Video file path, directory, or webcam ID for pose estimation')
    parser.add_argument(
        '--output-path', type=str, default='output',
        help='Output path for the results.')
    parser.add_argument(
        '--model', type=str,  default="ckpts/vitpose/vitpose-h-coco.pth",
        help='checkpoint path of the model')
    parser.add_argument(
        '--yolo', type=str, required=False,
        default="ckpts\\yolo\\yolo11x.pt",
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
        '--model-name', type=str, required=False, default="h",
        choices=['s', 'b', 'l', 'h'],
        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument(
        '--yolo-size', type=int, required=False, default=320,
        help='YOLO image size during inference')

    conf_threshold_help = (
        'Minimum confidence for keypoints to be drawn. [0, 1] range')
    parser.add_argument(
        '--conf-threshold', type=float, required=False, default=0.05,
        help=conf_threshold_help)

    parser.add_argument(
        '--single-pose', default=False, action='store_true',
        help='Do not use SORT tracker because single pose is expected')
    parser.add_argument(
        '--show', default=True, action='store_true',
        help='preview result during inference')
    parser.add_argument(
        '--show-yolo', default=True, action='store_true',
        help='draw yolo results')
    parser.add_argument(
        '--save-json', default=False, action='store_true',
        help='save json results')
    args = parser.parse_args()

    # --- Setup models and devices ---
    use_cuda = torch.cuda.is_available()
    yolo = args.yolo
    if yolo is None:
        yolo = 'vitpose/yolov8s.pt'

    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=True)
    print(f">>> Model loaded: {args.model}")

    # --- Get list of video files ---
    input_paths = []
    if os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                input_paths.append(os.path.join(args.input, filename))
        print(f">>> Found {len(input_paths)} videos in the directory.")
    elif os.path.isfile(args.input):
        input_paths.append(args.input)
    else:
        sys.exit(f"Error: Input path is not a valid file or directory: {args.input}")

    os.makedirs(args.output_path, exist_ok=True)

    # --- Main loop for each video ---
    for video_path in input_paths:
        print(f"\n>>> Processing video: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            continue

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # --- Setup Video Writer ---
        output_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(
            args.output_path, f"{output_filename}_result.mp4")
        
        # Output frame size will be half of the original
        out_size = (frame_width // 2, frame_height // 2)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, out_size)

        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                resized_frame = cv2.resize(frame_rgb, out_size)
                
                model.inference(resized_frame) 
                
                image_to_show = model.draw(
                    args.show_yolo, False, args.conf_threshold
                )[..., ::-1]

                out_writer.write(image_to_show)

                if args.show:
                    cv2.imshow('Keypoints', image_to_show)

            except Exception as e:
                print(f"Processing error on frame {frame_counter}: {e}")
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_counter += 1

        print(f">>> Finished processing. Output video saved to: {output_video_path}")
        cap.release()
        out_writer.release()

    cv2.destroyAllWindows()
    print("\n>>> All videos processed.")


if __name__ == "__main__":
    main()