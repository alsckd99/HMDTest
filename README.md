HMDTest 설치 가이드
이 문서는 HMDTest 프로젝트를 설정하고 실행하는 데 필요한 단계를 안내합니다.

설치 과정
1. Git 저장소 복제
먼저, 아래 명령어를 사용하여 프로젝트 파일을 로컬 컴퓨터로 복제합니다.
# HMDTest 설치 가이드

이 문서는 HMDTest 프로젝트를 설정하고 실행하는 데 필요한 단계를 안내합니다.

## 설치 과정

### 1. Git 저장소 복제
먼저, 아래 명령어를 사용하여 프로젝트 파일을 로컬 컴퓨터로 복제합니다.
```bash
git clone [https://github.com/alsckd99/HMDTest.git](https://github.com/alsckd99/HMDTest.git)

2. Conda 가상환경 생성
프로젝트 실행을 위한 별도의 가상환경을 만듭니다. Python 버전은 3.10으로 설정합니다.

Bash

conda create -n hmdtest python=3.10
생성 후에는 아래 명령어로 가상환경을 활성화해주세요.

Bash

conda activate hmdtest
3. CUDA 및 PyTorch 설치
머신러닝 모델 실행을 위해 CUDA와 PyTorch를 설치해야 합니다. 사용자의 그래픽카드 및 CUDA 버전에 맞는 설치 명령어는 PyTorch 공식 홈페이지에서 직접 확인 후 실행해주세요.

4. PoseEstimation 폴더로 이동
설치에 필요한 파일이 있는 PoseEstimation 폴더로 이동합니다.

Bash

cd HMDTest/PoseEstimation
5. 필요 라이브러리 설치
마지막으로, requirements.txt 파일에 명시된 모든 파이썬 라이브러리를 한 번에 설치합니다.

Bash

pip install -r requirements.txt
