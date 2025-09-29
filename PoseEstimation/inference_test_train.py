import argparse
import json
import os
import time
import glob

from PIL import Image
import cv2
import numpy as np
import torch
import tqdm

from vitpose.vit_utils.inference import NumpyEncoder, VideoReader
from vitpose.inference import VitInference


def inference(args):
    try:
        import onnxruntime  # noqa: F401
        has_onnx = True
    except ModuleNotFoundError:
        has_onnx = False

    use_mps = (hasattr(torch.backends, 'mps') and
               torch.backends.mps.is_available())
    use_cuda = torch.cuda.is_available()

    # Load Yolo
    yolo = args.yolo
    if yolo is None:
        yolo_suffix = '.onnx' if has_onnx and not (use_mps or use_cuda) \
            else '.pt'
        yolo = 'vitpose/yolov8s' + yolo_suffix
    input_path = args.input

    # Load the image / video reader
    try:  # Check if is webcam
        int(input_path)
        is_video = True
    except ValueError:
        assert os.path.isfile(input_path), 'The input file does not exist'
        is_video = input_path[input_path.rfind('.') + 1:].lower() in \
            ['avi', 'mp4', 'mov']

    ext = '.mp4' if is_video else '.png'
    if args.save_img or args.save_json:
        assert args.output_path, \
            'Specify an output path if using save-img or save-json flags'
    output_path = args.output_path

    # Output path
    file_output_path = os.path.join(output_path, os.path.basename(input_path))
    os.makedirs(file_output_path, exist_ok=True)
    og_ext = input_path[input_path.rfind('.'):]
    save_name_img = os.path.basename(input_path).replace(
        og_ext, f"_result{ext}")
    output_path_img = os.path.join(file_output_path, save_name_img)

    wait = 0
    total_frames = 1
    if is_video:
        reader = VideoReader(input_path, args.rotate)
        cap = cv2.VideoCapture(input_path)  # type: ignore
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        wait = 15
        if args.save_img:
            cap = cv2.VideoCapture(input_path)  # type: ignore
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            cap.release()
            assert ret
            assert fps > 0
            output_size = frame.shape[:2][::-1]

            # Check if we have X264 otherwise use default MJPG
            try:
                fourcc = cv2.VideoWriter_fourcc(*'h264')
                temp_video = cv2.VideoWriter('/tmp/checkcodec.mp4',
                                             fourcc, 30, (32, 32))
                opened = temp_video.isOpened()
            except Exception:
                opened = False
            codec = 'h264' if opened else 'MJPG'
            out_writer = cv2.VideoWriter(output_path_img,
                                         cv2.VideoWriter_fourcc(*codec),
                                         fps, output_size)  # type: ignore
    else:
        reader = [np.array(Image.open(input_path).rotate(args.rotate))]

    # Initialize model
    model = VitInference(args.model, yolo, args.model_name,
                         args.det_class, args.dataset,
                         args.yolo_size, is_video=is_video)
    print(f">>> Model loaded: {args.model}")

    print(f'>>> Running inference on {input_path}')
    
    output_annotations = []
    keypoints_list = []
    fps_list = []
    tot_time = 0.

    for (ith, img) in tqdm.tqdm(enumerate(reader), total=total_frames):
        t0 = time.time()

        # Check if the image is grayscale and convert to RGB if necessary
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Run inference
        frame_keypoints = model.inference(img)

        delta = time.time() - t0
        tot_time += delta
        fps_list.append(delta)

        if not is_video:
            try:
                image_id = int(os.path.splitext(
                    os.path.basename(input_path))[0])
            except ValueError:
                image_id = os.path.splitext(
                    os.path.basename(input_path))[0]
        else:
            image_id = ith
        
        bboxes, _, scores = model._tracker_res
        for i, kps in enumerate(frame_keypoints.values()):
            bbox = bboxes[i]
            score = scores[i]

            x1, y1, x2, y2 = bbox
            coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            flat_kps = []
            for kp in kps:
                x, y, conf = kp
                x_val, y_val = round(float(x), 2), round(float(y), 2)
                v = 0
                if conf > 0:
                    v = 1
                if conf > args.conf_threshold:
                    v = 2
                flat_kps.extend([x_val, y_val, v])

            annotation = {
                'image_id': image_id,
                'category_id': 1,
                'bbox': coco_bbox,
                'keypoints': flat_kps,
                'score': float(score)
            }
            output_annotations.append(annotation)

        # Draw the poses and save the output img
        if args.show or args.save_img:
            # Draw result and transform to BGR
            img = model.draw(args.show_yolo, args.show_raw_yolo,
                             args.conf_threshold)[..., ::-1]

            if args.save_img:
                if is_video:
                    out_writer.write(img)
                else:
                    print('>>> Saving output image')
                    cv2.imwrite(output_path_img, img)

            if args.show:
                cv2.imshow('preview', img)
                cv2.waitKey(wait)

    if is_video:
        tot_poses = sum(len(k) for k in keypoints_list)
        print(f'>>> Mean inference FPS: {1 / np.mean(fps_list):.2f}')
        print(f'>>> Total poses predicted: {tot_poses} mean per frame: '
              f'{(tot_poses / (ith + 1)):.2f}')
        print(f'>>> Mean FPS per pose: {(tot_poses / tot_time):.2f}')

    if is_video and args.save_img:
        out_writer.release()
    cv2.destroyAllWindows()
    return output_annotations


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='path to image / video or webcam ID (=cv2)')
    parser.add_argument('--output-path', type=str, default='',
                        help='Output path for the results.')
    parser.add_argument('--model', type=str, required=True,
                        help='checkpoint path of the model')
    parser.add_argument('--yolo', type=str, required=False,
                        default="ckpts\\yolo\\yolo11m.pt",
                        help='checkpoint path of the yolo model')
    parser.add_argument(
        '--dataset', type=str, required=False, default=None,
        help='Name of the dataset. If None it is extracted from the file name.'
             '["coco", "coco_25", "wholebody", "mpii", '
             '"ap10k", "apt36k", "aic", "custom"]')
    parser.add_argument(
        '--det-class', type=str, required=False, default=None,
        help='["human", "cat", "dog", "horse", "sheep", '
             '"cow", "elephant", "bear", "zebra", "giraffe", "animals"]')
    parser.add_argument('--model-name', type=str, required=False,
                        choices=['s', 'b', 'l', 'h'],
                        help='[s: ViT-S, b: ViT-B, l: ViT-L, h: ViT-H]')
    parser.add_argument('--yolo-size', type=int, required=False, default=320,
                        help='YOLO image size during inference')
    parser.add_argument(
        '--conf-threshold', type=float, required=False, default=0.5,
        help='Minimum confidence for keypoints to be drawn. [0, 1] range')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270],
                        required=False, default=0,
                        help='Rotate the image of [90, 180, 270] '
                        'degress counterclockwise')
    parser.add_argument(
        '--single-pose', default=False, action='store_true',
        help='Do not use SORT tracker because single pose is expected')
    parser.add_argument('--show', default=False, action='store_true',
                        help='preview result during inference')
    parser.add_argument('--show-yolo', default=False, action='store_true',
                        help='draw yolo results')
    parser.add_argument(
        '--show-raw-yolo', default=False, action='store_true',
        help=('draw yolo result before that SORT is applied for tracking'
              ' (only valid during video inference)'))
    parser.add_argument('--save-img', default=False, action='store_true',
                        help='save image results')
    parser.add_argument('--save-json', default=False, action='store_true',
                        help='save json results')
    args = parser.parse_args()

    all_annotations = []
    # If the input is a folder
    if os.path.isdir(args.input):
        img_files = glob.glob(os.path.join(args.input, '*'))
        img_files = [f for f in img_files if f.lower().endswith(
            ('.png', '.jpg', '.jpeg'))]
        assert img_files, 'No image files found in the directory'

        # Run inference on each video file
        for file in img_files:
            # Perform inference on each video file
            print(f">>> Running inference on image: {file}")
            args.input = file
            all_annotations.extend(inference(args))
    else:
        all_annotations.extend(inference(args))

    if args.save_json:
        output_json_path = ""
        if os.path.isdir(args.input):
            if args.output_path:
                output_json_path = os.path.join(
                    args.output_path, 'result.json')
                os.makedirs(args.output_path, exist_ok=True)
        else:
            if args.output_path:
                og_ext = args.input[args.input.rfind('.'):]
                save_name_json = os.path.basename(
                    args.input).replace(og_ext, "_result.json")
                file_output_path = os.path.join(
                    args.output_path,
                    os.path.splitext(os.path.basename(args.input))[0])
                os.makedirs(file_output_path, exist_ok=True)
                output_json_path = os.path.join(
                    file_output_path, save_name_json)

        if output_json_path:
            print(f'>>> Saving output json to {output_json_path}')
            with open(output_json_path, 'w') as f:
                json.dump(all_annotations, f, cls=NumpyEncoder)