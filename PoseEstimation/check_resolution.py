import cv2

def check_webcam_resolutions(device_id=1):
    """
    웹캠에서 지원하는 해상도를 확인하기 위해 일반적인 해상도 목록을 테스트하고
    실제로 설정된 값을 출력합니다.
    """
    # 사용자의 코드에 맞게 CAP_DSHOW를 사용합니다.
    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"오류: 비디오 장치 {device_id}를 열 수 없습니다.")
        return

    # 테스트해 볼 일반적인 해상도 목록
    resolutions = [
        (640, 480),
        (800, 600),
        (1280, 720),
        (1280, 960),  # 문제가 된 해상도
        (1920, 1080),
    ]

    print(f"--- 웹캠 {device_id}에서 지원하는 해상도 확인 ---")
    supported_resolutions = set()

    for w, h in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        
        # 실제로 설정된 해상도 값을 다시 가져옵니다.
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"요청: {w}x{h}, 실제 설정: {actual_w}x{actual_h}")
        supported_resolutions.add((actual_w, actual_h))
    
    print("\n--- 확인된 고유 지원 해상도 목록 ---")
    # 정렬해서 보기 좋게 출력
    for res in sorted(list(supported_resolutions), key=lambda x: x[0]):
        print(f"{res[0]}x{res[1]}")

    cap.release()

if __name__ == "__main__":
    # 사용하시는 웹캠 ID가 1이므로 1로 설정합니다.
    # 만약 다른 카메라를 확인하고 싶다면 이 숫자를 바꾸세요.
    check_webcam_resolutions(1)
