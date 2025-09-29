import abc
import os
from typing import Optional
import typing
import concurrent.futures

import cv2
import numpy as np
import torch
import torch.quantization

from ultralytics import YOLO

from .configs.ViTPose_common import data_cfg
from .sort import Sort
from .vit_models.model import ViTPose
from .vit_utils.inference import draw_bboxes, pad_image
from .vit_utils.top_down_eval import keypoints_from_heatmaps
from .vit_utils.util import dyn_model_import, infer_dataset_by_path
from .vit_utils.visualization import draw_points_and_skeleton, joints_dict

try:
    import torch_tensorrt
except ModuleNotFoundError:
    pass

try:
    import onnxruntime
except ModuleNotFoundError:
    pass

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError as e:
    print(e)
    TENSORRT_AVAILABLE = False

__all__ = ['VitInference']
np.bool = np.bool_
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


DETC_TO_YOLO_YOLOC = {
    'human': [0],
    'cat': [15],
    'dog': [16],
    'horse': [17],
    'sheep': [18],
    'cow': [19],
    'elephant': [20],
    'bear': [21],
    'zebra': [22],
    'giraffe': [23],
    'animals': [15, 16, 17, 18, 19, 20, 21, 22, 23]
}


class VitInference:

    def __init__(self, model: str,
                 yolo: str,
                 model_name: Optional[str] = None,
                 det_class: Optional[str] = None,
                 dataset: Optional[str] = None,
                 yolo_size: Optional[int] = 320,
                 device: Optional[str] = None,
                 is_video: Optional[bool] = False):

        # Device priority is cuda / mps / cpu
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.device = device
        self.yolo = YOLO(yolo)
        self.yolo_size = yolo_size
        self.is_video = is_video

        # State saving during inference
        self.save_state = True  # Can be disabled manually
        self._img = None
        self._yolo_res = None
        self._tracker_res = None
        self._keypoints = None
        
        # ID 매핑 시스템
        self.id_mapping = {}  # {tracker_id: sequential_id}
        self.next_sequential_id = 0
        self.active_ids = set()  # 현재 활성화된 sequential ID들

        # Use extension to decide which kind of model has been loaded
        use_onnx = model.endswith('.onnx')
        use_trt = model.endswith('.engine')
        

        # Extract dataset name
        if dataset is None:
            dataset = infer_dataset_by_path(model)

        assert dataset in ['mpii', 'coco', 'coco_25', 'wholebody', 'aic', 'ap10k', 'apt36k'], \
            'The specified dataset is not valid'

        # Dataset can now be set for visualization
        self.dataset = dataset

        # if we picked the dataset switch to correct yolo classes if not set
        if det_class is None:
            det_class = 'animals' if dataset in ['ap10k', 'apt36k'] else 'human'
        self.yolo_classes = DETC_TO_YOLO_YOLOC[det_class]

        assert model_name in [None, 's', 'b', 'l', 'h'], \
            f'The model name {model_name} is not valid'

        # onnx / trt models do not require model_cfg specification
        if model_name is None:
            assert use_onnx or use_trt, \
                'Specify the model_name if not using onnx / trt'
        else:
            # Dynamically import the model class
            model_cfg = dyn_model_import(self.dataset, model_name)

        self.target_size = data_cfg['image_size']
        if use_onnx:
            self._ort_session = onnxruntime.InferenceSession(model,
                                                             providers=['CUDAExecutionProvider',
                                                                        'CPUExecutionProvider'])
            inf_fn = self._inference_onnx
        else:
            self._vit_pose = ViTPose(model_cfg)
            self._vit_pose.eval()

            if use_trt:
                # TensorRT 엔진 로드
                with open(model, 'rb') as f:
                    engine_data = f.read()
                
                # TensorRT 런타임 생성
                logger = trt.Logger(trt.Logger.INFO)
                runtime = trt.Runtime(logger)
                
                # 엔진 역직렬화
                self._trt_engine = runtime.deserialize_cuda_engine(engine_data)
                
                # 실행 컨텍스트 생성
                self._trt_context = self._trt_engine.create_execution_context()
                
                # 입력 및 출력 바인딩 설정
                num_tensors = self._trt_engine.num_io_tensors
                binding_map = {}
                for i in range(num_tensors):
                    name = self._trt_engine.get_tensor_name(i)
                    binding_map[name] = i

                # 원하는 이름의 인덱스를 꺼내 쓰기
                self._trt_input_idx  = binding_map["input"]
                self._trt_output_idx = binding_map["output"]
                
                # 추론 함수 설정
                inf_fn = self._inference_tensorrt
            else:
                ckpt = torch.load(model, map_location='cpu')
                if 'state_dict' in ckpt:
                    self._vit_pose.load_state_dict(ckpt['state_dict'])
                else:
                    self._vit_pose.load_state_dict(ckpt)
                
                # 모델을 지정된 디바이스로 이동
                self._vit_pose = self._vit_pose.to(self.device)
                inf_fn = self._inference_torch

        # Override _inference abstract with selected engine
        self._inference = inf_fn  # type: ignore
    
    def _get_sequential_id(self, tracker_id):
        """트래커 ID를 순차적인 ID로 매핑"""
        if tracker_id in self.id_mapping:
            return self.id_mapping[tracker_id]
        
        # 새로운 ID 할당 - 가장 작은 빈 번호 찾기
        available_id = 0
        while available_id in self.active_ids:
            available_id += 1
        
        self.id_mapping[tracker_id] = available_id
        self.active_ids.add(available_id)
        return available_id
    
    def _update_active_ids(self, current_tracker_ids):
        """현재 프레임의 트래커 ID들을 기반으로 활성 ID 업데이트"""
        # 현재 프레임에 없는 트래커 ID들 제거
        inactive_tracker_ids = set(self.id_mapping.keys()) - set(current_tracker_ids)
        for tracker_id in inactive_tracker_ids:
            if tracker_id in self.id_mapping:
                sequential_id = self.id_mapping[tracker_id]
                self.active_ids.discard(sequential_id)
                del self.id_mapping[tracker_id]

    @classmethod
    def postprocess(cls, heatmaps, org_w, org_h):
        points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                               center=np.array([[org_w // 2,
                                                                 org_h // 2]]),
                                               scale=np.array([[org_w, org_h]]),
                                               unbiased=True, use_udp=True)
        return np.concatenate([points[:, :, ::-1], prob], axis=2)

    @abc.abstractmethod
    def _inference(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def tracking(self, img: np.ndarray) -> dict[typing.Any, typing.Any]:
        # 소켓 버퍼 크기 제한 문제를 방지하기 위해 이미지 크기 축소
        scale_factor = 0.8
        img_resized = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        results = self.yolo.track(img_resized[..., ::-1], tracker="bytetrack.yaml", 
                                  persist=True, imgsz=self.yolo_size, classes=0, 
                                  device=self.device if self.device != 'cuda' else 0)
        # 결과 이미지 압축을 위해 품질 낮추기
        is_success, annotated_img = cv2.imencode('.jpg', results[0].plot(), [cv2.IMWRITE_JPEG_QUALITY, 85])
        if is_success:
            return cv2.imdecode(annotated_img, cv2.IMREAD_COLOR)
        return results[0].plot()
    def inference(self, img: np.ndarray) -> dict[typing.Any, typing.Any]:
        res_pd = np.empty((0, 4))
        results = None
        tracker_ids = None
        frame_keypoints = []
        scores_bbox = {}
        
        results = self.yolo.track(img[..., ::-1],tracker="bytetrack.yaml", persist=True, imgsz = self.yolo_size, classes=0, device=self.device if self.device != 'cuda' else 0)
        anotated_frame = results[0].plot()
        res_pd = np.array([r[:4].tolist()+[r[5]] for r in results[0].boxes.data.cpu().numpy() if r[5]>0.8]).reshape(-1,5)
        # ID가 없는 경우를 처리합니다
        if results[0].boxes.id is None:
            tracker_ids = list(range(len(results[0].boxes.data)))
        else:
            tracker_ids = results[0].boxes.id.int().cpu().tolist()
            
        # 현재 트래커 ID들을 기반으로 활성 ID 업데이트
        self._update_active_ids(tracker_ids)
        
        # Prepare boxes for inference
        bboxes = res_pd[:, :4].round().astype(int)
        scores = res_pd[:, 4].tolist()
        pad_bbox = 10
        
        # 순차적인 ID로 매핑하여 처리
        sequential_ids = []
        for tracker_id in tracker_ids:
            sequential_id = self._get_sequential_id(tracker_id)
            sequential_ids.append(sequential_id)
        
        for bbox, tracker_id, sequential_id in zip(bboxes, tracker_ids, 
                                                   sequential_ids):
            bbox[[0, 2]] = np.clip(bbox[[0, 2]] + [-pad_bbox, pad_bbox], 
                                   0, img.shape[1])
            bbox[[1, 3]] = np.clip(bbox[[1, 3]] + [-pad_bbox, pad_bbox], 
                                   0, img.shape[0])

            img_inf = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

            keypoints = self._inference(img_inf)[0]
            keypoints[:, :2] += bbox[:2][::-1] - [top_pad, left_pad]
            frame_keypoints.append(keypoints)
            
            # 순차적인 ID로 점수 저장
            tracker_idx = tracker_ids.index(tracker_id)
            if tracker_idx < len(scores):
                scores_bbox[sequential_id] = scores[tracker_idx]
                
        try:
            frame_keypoints[-1][9] = ((np.array(frame_keypoints[-1][8])+np.array(frame_keypoints[-1][9]))*0.5).tolist()
        except Exception as e:
            print(e)
            pass
            
        if self.save_state:
            self._img = img
            self._yolo_res = results[0].plot()
            self._tracker_res = (bboxes, sequential_ids, scores)
            self._keypoints = frame_keypoints
            self._scores_bbox = scores_bbox

        return frame_keypoints, anotated_frame
    
    # def inference(self, img: np.ndarray) -> dict[typing.Any, typing.Any]:
    #     res_pd = np.empty((0, 4))
    #     results = None
    #     track_ids = None
    #     frame_keypoints = []
    #     scores_bbox = {}
        
    #     results = self.yolo.track(img[..., ::-1],tracker="bytetrack.yaml", persist=True, imgsz = self.yolo_size, classes=0, device=self.device if self.device != 'cuda' else 0)
    #     anotated_frame = results[0].plot()
    #     res_pd = np.array([r[:4].tolist()+[r[5]] for r in results[0].boxes.data.cpu().numpy() if r[5]>0.35]).reshape(-1,5)
    #     # ID가 없는 경우를 처리합니다
    #     if results[0].boxes.id is None:
    #         track_ids = range(len(results[0].boxes.data))
    #     else:
    #         track_ids = range(len(results[0].boxes.id.int().cpu().tolist()))
    #     # Prepare boxes for inference
    #     bboxes = res_pd[:, :4].round().astype(int)
    #     scores = res_pd[:, 4].tolist()
    #     pad_bbox = 10

    #     if track_ids is None:
    #         track_ids = range(len(bboxes))
            
    #     # 스레드 풀을 사용한 병렬 처리
    #     def process_bbox(args):
    #         bbox, id = args
    #         bbox_copy = bbox.copy()
    #         bbox_copy[[0, 2]] = np.clip(bbox_copy[[0, 2]] + [-pad_bbox, pad_bbox], 0, img.shape[1])
    #         bbox_copy[[1, 3]] = np.clip(bbox_copy[[1, 3]] + [-pad_bbox, pad_bbox], 0, img.shape[0])

    #         img_inf = img[bbox_copy[1]:bbox_copy[3], bbox_copy[0]:bbox_copy[2]]
    #         img_inf, (left_pad, top_pad) = pad_image(img_inf, 3 / 4)

    #         keypoints = self._inference(img_inf)[0]
    #         keypoints[:, :2] += bbox_copy[:2][::-1] - [top_pad, left_pad]
            
    #         score = scores[id] if 0 <= id < len(scores) else 0.0
    #         return keypoints, id, score
        
    #     # 스레드 풀 생성 및 실행
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(bboxes))) as executor:
    #         thread_results = list(executor.map(process_bbox, zip(bboxes, track_ids)))
            
    #     # 결과 정리
    #     for keypoints, id, score in thread_results:
    #         frame_keypoints.append(keypoints)
    #         if score > 0:
    #             scores_bbox[id] = score
                
    #     try:
    #         if frame_keypoints:
    #             frame_keypoints[-1][9] = ((np.array(frame_keypoints[-1][8])+np.array(frame_keypoints[-1][9]))*0.5).tolist()
    #     except Exception as e:
    #         print(e)
    #         pass
    #     if self.save_state:
    #         self._img = img
    #         self._yolo_res = results[0].plot()
    #         self._tracker_res = (bboxes, track_ids, scores)
    #         self._keypoints = frame_keypoints
    #         self._scores_bbox = scores_bbox

    #     return frame_keypoints, anotated_frame

    def draw(self, show_yolo=False, show_raw_yolo=False, confidence_threshold=0.5):
        img = self._img.copy()
        bboxes, ids, scores = self._tracker_res

        if self._yolo_res is not None and (show_raw_yolo or show_yolo):
            img = np.array(self._yolo_res)[..., ::-1]
            
        img = np.array(img)[..., ::-1]  # RGB to BGR for cv2 modules
        
        # 키포인트가 비어있는 경우 원본 이미지 반환
        if not self._keypoints or len(self._keypoints) == 0:
            return img[..., ::-1]  # Return RGB as original
        for idx, k in enumerate(np.array(self._keypoints)):
            img = draw_points_and_skeleton(img.copy(), k,
                                           joints_dict()[self.dataset]['skeleton'],
                                           person_index=idx,
                                           points_color_palette='gray',
                                           skeleton_color_palette='jet',
                                           points_palette_samples=10,
                                           confidence_threshold=confidence_threshold)
        return img[..., ::-1]  # Return RGB as original

    def pre_img(self, img):
        org_h, org_w = img.shape[:2]
        img_input = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR) / 255
        img_input = ((img_input - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return img_input, org_h, org_w

    @torch.no_grad()
    def _inference_torch(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)
        img_input = torch.from_numpy(img_input).to(torch.device(self.device))

        # Feed to model
        heatmaps = self._vit_pose(img_input).detach().cpu().numpy()
        return self.postprocess(heatmaps, org_w, org_h)

    def _inference_onnx(self, img: np.ndarray) -> np.ndarray:
        # Prepare input data
        img_input, org_h, org_w = self.pre_img(img)

        # Feed to model
        ort_inputs = {self._ort_session.get_inputs()[0].name: img_input}
        heatmaps = self._ort_session.run(None, ort_inputs)[0]
        return self.postprocess(heatmaps, org_w, org_h)
    
    def normalize_keypoints(self,keypoints, angle):
        theta = np.deg2rad(angle)

        # y축을 기준으로 회전하는 3D 회전 행렬 (간단화)
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        # 3D 키포인트로 변환 (z=0)
        keypoints_3d = np.hstack((keypoints, np.zeros((keypoints.shape[0], 1))))
        # 회전 적용
        rotated_keypoints = keypoints_3d.dot(R.T)
        # 투영 적용 (단순 원근 투영)
        f = 1  # 초점 거리 (적절히 설정)
        projected_keypoints = rotated_keypoints[:, :2] / (rotated_keypoints[:, 2].reshape(-1, 1) + 1e-6)
        return projected_keypoints

    def _inference_tensorrt(self, img: np.ndarray) -> np.ndarray:
        """TensorRT 엔진을 사용한 추론 함수"""
        # 이미지 전처리
        img_input, org_h, org_w = self.pre_img(img)
        # 입력 데이터 준비
        input_tensor = torch.from_numpy(img_input).contiguous()
        
        # 입력 텐서 형태 가져오기
        input_shape = input_tensor.shape
        
        # CUDA 메모리 할당
        d_input = cuda.mem_alloc(input_tensor.numel() * input_tensor.element_size())
        
        # 출력 크기 확인 - 최신 TensorRT API 사용
        output_size = 1
        for dim in self._trt_engine.get_tensor_shape("output"):
            if dim > 0:
                output_size *= dim
                
        # 출력 메모리 할당
        h_output = cuda.pagelocked_empty(output_size, dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)
        
        # 호스트에서 디바이스로 데이터 복사
        cuda.memcpy_htod(d_input, input_tensor.cpu().numpy())
        
        # 입력 텐서 차원 명시적으로 설정 (중요!)
        self._trt_context.set_input_shape("input", input_shape)
        
        # 바인딩 설정 - 최신 API 방식으로 수정
        self._trt_context.set_tensor_address("input", int(d_input))
        self._trt_context.set_tensor_address("output", int(d_output))
        
        # 추론 실행
        stream = cuda.Stream()
        self._trt_context.execute_async_v3(stream.handle)
        stream.synchronize()
        
        # 결과 디바이스에서 호스트로 복사
        cuda.memcpy_dtoh(h_output, d_output)
        
        # 결과 재구성
        output_shape = self._trt_engine.get_tensor_shape("output")
        heatmaps = h_output.reshape(output_shape)
        
        return self.postprocess(heatmaps, org_w, org_h)