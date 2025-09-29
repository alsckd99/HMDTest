import os
import argparse
import torch
import numpy as np
from pathlib import Path

from vit_models.model import ViTPose
from vit_utils.util import dyn_model_import, infer_dataset_by_path

# TensorRT 관련 라이브러리 import
try:
    import tensorrt as trt
    from cuda import cudart
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError as e:
    print(e)
    # print("TensorRT 또는 CUDA 라이브러리가 설치되어 있지 않습니다.")
    # print("pip install nvidia-tensorrt pycuda 명령으로 설치하세요.")
    exit(1)

def convert_to_onnx(model, input_shape, output_path):
    """PyTorch 모델을 ONNX 형식으로 변환합니다."""
    print(f"ONNX 변환 중: {output_path}")
    
    # 더미 입력 생성
    dummy_input = torch.randn(*input_shape, device="cpu")
    
    # ONNX 내보내기
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"ONNX 모델이 저장되었습니다: {output_path}")
    return output_path

def onnx_to_tensorrt(onnx_path, engine_path, precision="fp16", max_workspace_size=1<<30, batch_size=1):
    """ONNX 모델을 TensorRT 엔진으로 변환합니다."""
    print(f"TensorRT 엔진 생성 중: {engine_path}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # TensorRT builder 생성
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    # 정밀도 설정
    if precision.lower() == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 모드로 변환합니다.")
    elif precision.lower() == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("INT8 모드로 변환합니다. (캘리브레이션이 필요할 수 있습니다)")
    else:
        print("FP32 모드로 변환합니다.")
    
    # ONNX 파서 생성 및 파싱
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(f"ONNX 파싱 오류: {parser.get_error(error)}")
            return None
    
    # 배치 크기 설정
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 3, 256, 192), (batch_size, 3, 256, 192), (batch_size*2, 3, 256, 192))
    config.add_optimization_profile(profile)
    
    # 엔진 직렬화
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print("TensorRT 엔진 생성 실패")
        return None
    
    # 엔진 저장
    with open(engine_path, "wb") as f:
        f.write(engine)
    
    print(f"TensorRT 엔진이 저장되었습니다: {engine_path}")
    return engine_path

def load_vitpose_model(model_path, model_name=None, dataset=None):
    """ViTPose 모델을 로드합니다."""
    # 모델 이름이 지정되지 않은 경우 추론
    if model_name is None:
        assert model_path.endswith('.pth'), "모델 이름이 지정되지 않은 경우 .pth 파일이어야 합니다."
    
    # 데이터셋 이름이 지정되지 않은 경우 추론
    if dataset is None:
        dataset = infer_dataset_by_path(model_path)
    
    # 모델 구성 가져오기
    model_cfg = dyn_model_import(dataset, model_name)
    
    # 모델 초기화 및 가중치 로드
    model = ViTPose(model_cfg)
    model.eval()
    
    # 체크포인트 로드
    ckpt = torch.load(model_path, map_location='cpu')
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    return model, dataset

def main():
    parser = argparse.ArgumentParser(description="PyTorch ViTPose 모델을 ONNX 및 TensorRT로 변환")
    parser.add_argument("--model", type=str, required=True, help="PyTorch 모델 경로 (.pth)")
    parser.add_argument("--model_name", type=str, default=None, choices=[None, 's', 'b', 'l', 'h'], 
                        help="모델 크기 (s, b, l, h)")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="데이터셋 이름 (mpii, coco, coco_25, wholebody, aic, ap10k, apt36k)")
    parser.add_argument("--output_dir", type=str, default="../ckpts/vitpose/trt_models", help="출력 디렉토리")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "int8"], 
                        help="TensorRT 정밀도 모드")
    parser.add_argument("--batch_size", type=int, default=1, help="배치 크기")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 모델 로드
    model, dataset = load_vitpose_model(args.model, args.model_name, args.dataset)
    print(f"모델 로드 완료: {args.model}, 데이터셋: {dataset}")
    
    # 파일 이름 생성
    base_name = Path(args.model).stem
    output_base = f"{base_name}_{dataset}"
    onnx_path = os.path.join(args.output_dir, f"{output_base}.onnx")
    engine_path = os.path.join(args.output_dir, f"{output_base}_{args.precision}.engine")
    
    # 입력 크기 설정 (B, C, H, W)
    input_shape = (args.batch_size, 3, 256, 192)  # 기본 ViTPose 입력 크기
    
    # ONNX로 변환
    onnx_path = convert_to_onnx(model, input_shape, onnx_path)
    
    # TensorRT로 변환
    engine_path = onnx_to_tensorrt(onnx_path, engine_path, args.precision, batch_size=args.batch_size)
    
    if engine_path:
        print("\n변환 프로세스가 성공적으로 완료되었습니다.")
        print(f"ONNX 모델: {onnx_path}")
        print(f"TensorRT 엔진: {engine_path}")
        print("\n사용 방법:")
        print(f"1. VitInference 클래스에 TensorRT 엔진 경로를 전달하세요:")
        print(f"   model = VitInference(model='{engine_path}', yolo='yolov8n.pt', model_name=None, dataset='{dataset}')")
    else:
        print("변환 실패")

if __name__ == "__main__":
    main() 