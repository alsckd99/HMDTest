import matplotlib.pyplot as plt
import numpy as np

# 정규화된 좌표값 (x, y)
coordinates = {
    'nose': [353.30255255126953125/640, 409.89780818359375/480],
    'left_eye': [345.2271483339844/640, 418.02825927734375/480],
    'right_eye': [362.06732177734375/640, 417.015380859375/480],
    'left_shoulder': [305.2596435546875/640, 346.98478729296875/480],
    'right_shoulder': [388.87258935546875/640, 342.78478729296875/480],
    'left_elbow': [285.04058837890625/640, 275.4458312988281/480],
    'right_elbow': [399.9532470703125/640,269.8273620605469/480],
    'left_wrist': [284.05377197265625/640, 211.64967346191406/480],
    'right_wrist': [398.3096618652344/640, 205.17184448242188/480],
    'left_hip': [313.157353515625/640, 209.6177421875/480],
    'right_hip': [364.93145751953125/640, 205.59054565429688/480],
    'left_knee': [303.87353515625/640, 118.10107421875/480],
    'right_knee': [359.21881103515625/640, 113.90771484375/480],
    'left_ankle': [300.37078857421875/640, 41.58544921875/480],
    'right_ankle': [351.4083251953125/640, 41.28839111328125/480],
    'middle_hip': [339.0445556640625/640, 207.6090850830078/480]
}
# coordinates = {
#     'nose': [0.517, 0.847],
#     'left_eye': [0.504, 0.864],
#     'right_eye': [0.528, 0.864],
#     'left_shoulder': [0.447, 0.714],
#     'right_shoulder': [0.536, 0.713],
#     'left_elbow': [0.418, 0.560],
#     'right_elbow': [0.603, 0.560],
#     'left_wrist': [0.420, 0.424],
#     'right_wrist': [0.599, 0.420],
#     'left_hip': [0.472, 0.423],
#     'right_hip': [0.500, 0.426],
#     'left_knee': [0.472, 0.361],
#     'right_knee': [0.552, 0.362],
#     'left_ankle': [0.471, 0.091],
#     'right_ankle': [0.550, 0.090],
#     'middle_hip': [0.627, 0.481]
# }


# 연결할 관절 쌍 정의
connections = [
    ('left_eye', 'right_eye'),
    ('left_eye', 'nose'),
    ('right_eye', 'nose'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'middle_hip'),
    ('right_hip', 'middle_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle')
]

plt.figure(figsize=(8, 8))
# y축 반전 (matplotlib에서는 y축이 위로 갈수록 증가하므로)
plt.gca().invert_yaxis()

# 관절 연결선 그리기
for connection in connections:
    start = coordinates[connection[0]]
    end = coordinates[connection[1]]
    plt.plot([start[0], end[0]], [start[1], end[1]], 'b-', linewidth=1)

# 관절 포인트 그리기
for joint, pos in coordinates.items():
    plt.plot(pos[0], pos[1], 'ro', markersize=5)

plt.title('Normalized Pose Visualization')
plt.xlabel('Normalized X')
plt.ylabel('Normalized Y')
plt.grid(True)
plt.show()