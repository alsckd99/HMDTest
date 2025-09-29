using UnityEngine;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEngine.Video;


public class JointDataExtractor : MonoBehaviour
{
    
    // --- Python 및 흐름 제어 ---
    private System.Diagnostics.Process pythonProcess;
    private string pythonScriptDirectory;
    private bool isCollectingData = false;
    private string startSignalFilePath;

    private float TARGET_FPS = 15.0f;
    private float timeBetweenFrames;
    private float timeSinceLastCapture = 0.0f;

    public UDPReceiver udpReceiver;
    
    // 현재 처리된 객체 데이터를 저장하는 변수들
    private ObjectData currentObjectData_0;
    private ObjectData currentObjectData_1;
    private ObjectData currentObjectData_2;
    
    // JointData 참조
    private JointData jointData;

    void Start()
    {
        timeBetweenFrames = 1.0f / TARGET_FPS;
        jointData = GetComponent<JointData>(); // JointData 참조 획득
        StartPython();
    }

    public void SetTargetFPS(float fps)
    {
        TARGET_FPS = fps;
        timeBetweenFrames = 1.0f / TARGET_FPS;
    }
    
    // JointData에서 객체 데이터에 접근할 수 있는 메서드
    public ObjectData GetObjectData(int id)
    {
        switch (id)
        {
            case 0:
                return currentObjectData_0;
            case 1:
                return currentObjectData_1;
            case 2:
                return currentObjectData_2;
            default:
                return null;
        }
    }
    void Update()
    {
        // GameObject가 비활성화되었는지 체크
        if (!gameObject.activeInHierarchy)
        {
            UnityEngine.Debug.LogWarning($"GameObject가 비활성화됨: {gameObject.name}");
            return;
        }

        // UDP 데이터 수신 플래그만 체크 (처리는 LateUpdate에서)
        if (udpReceiver.keypoints_dataReceived)
        {
            // 데이터가 수신되었다는 것만 확인
        }
    }

    private void ProcessMultipleObjects()
    {
        if (udpReceiver.keypoints?.objects == null || udpReceiver.keypoints.objects.Length == 0)
        {
            Debug.LogWarning("수신된 객체 데이터가 없습니다.");
            return;
        }

        Debug.Log($"총 {udpReceiver.keypoints.objects.Length}개의 객체가 감지되었습니다.");

        // 각 객체별로 데이터 처리
        for (int i = 0; i < udpReceiver.keypoints.objects.Length; i++)
        {
            ObjectData objData = udpReceiver.keypoints.objects[i];
            if (objData == null)
            {
                Debug.LogWarning($"Object {i}: 데이터가 null입니다.");
                continue;
            }

            // ID에 따라 해당 ObjectData에 할당
            switch (objData.id)
            {
                case 0:
                    currentObjectData_0 = objData;
                    //Debug.Log($"Object ID {objData.id} 데이터 할당 완료");
                    //currentObjectData_0.PrintObjectData();
                    break;
                case 1:
                    currentObjectData_1 = objData;
                    //Debug.Log($"Object ID {objData.id} 데이터 할당 완료");
                    //currentObjectData_1.PrintObjectData();
                    break;
                case 2:
                    currentObjectData_2 = objData;
                    //Debug.Log($"Object ID {objData.id} 데이터 할당 완료");
                    //currentObjectData_2.PrintObjectData();
                    break;
                default:
                    Debug.LogWarning($"Object ID {objData.id}: 처리할 수 없는 ID (ID 0-2만 지원)");
                    break;
            }
        }
        
        // 데이터 처리 완료 후 JointData에 알림
        if (jointData != null)
        {
            jointData.OnObjectDataUpdated();
        }
    }

    // 디버그용 변수들
    private float debugTimer = 0.0f;
    private int frameCount = 0;
    private float lastProcessTime = 0.0f;

    void LateUpdate()
    {
        timeSinceLastCapture += Time.deltaTime;
        debugTimer += Time.deltaTime;
        
        // 1초마다 실제 처리 빈도 출력
        if (debugTimer >= 1.0f)
        {
            float actualFPS = frameCount / debugTimer;
            Debug.Log($"[DEBUG] 설정 FPS: {TARGET_FPS}, 실제 처리 빈도: {actualFPS:F2}fps, 마지막 처리: {Time.time - lastProcessTime:F3}초 전");
            debugTimer = 0.0f;
            frameCount = 0;
        }
        
        if (timeSinceLastCapture < timeBetweenFrames) return;
        
        timeSinceLastCapture -= timeBetweenFrames;
        frameCount++;
        lastProcessTime = Time.time;
        

        // 프레임 수에 맞춰서 ObjectData 할당 처리
        if (udpReceiver.keypoints_dataReceived)
        {
            ProcessMultipleObjects();
            udpReceiver.keypoints_dataReceived = false; // 처리 완료 후 플래그 리셋
        }
        else
        {
            Debug.Log($"[DEBUG] UDP 데이터 없음 - 대기 중");
        }

        if (!isCollectingData)
        {
            if (!string.IsNullOrEmpty(startSignalFilePath) && File.Exists(startSignalFilePath) && File.ReadAllText(startSignalFilePath).Trim() == "start")
            {
                isCollectingData = true;
                File.Delete(startSignalFilePath);
                Debug.Log("C#: Python으로부터 시작 신호를 받아 데이터 수집을 시작합니다.");
            }
            else return;
        }
    }

    private void CleanupAndQuit()
    {
        QuitApplication();
    }

    void OnApplicationQuit()
    {
        StopPython();
    }

    void OnDestroy()
    {
    }

    private void QuitApplication()
    {
        Debug.Log("애플리케이션을 종료합니다.");
#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#else
        Application.Quit();
#endif
    }

    public void StartPython()
    {
        if (pythonProcess != null)
        {
            Debug.LogWarning("Python 스크립트가 이미 실행 중입니다.");
            return;
        }

        string projectPath = Application.dataPath.Replace("/Assets", "").Replace("\\Assets", "");
        string scriptPath = System.IO.Path.Combine(projectPath, "PoseEstimation","inference_tracker.py");
        this.pythonScriptDirectory = Path.GetDirectoryName(scriptPath);
        System.Diagnostics.ProcessStartInfo startInfo = new System.Diagnostics.ProcessStartInfo
        {
            FileName = "C:/Users/user/anaconda3/envs/hmdtest/python.exe",
            Arguments = $"-u \"{scriptPath}\"",
            WorkingDirectory = this.pythonScriptDirectory,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        pythonProcess = new System.Diagnostics.Process
        {
            StartInfo = startInfo,
            EnableRaisingEvents = true
        };

        pythonProcess.OutputDataReceived += (sender, args) =>
        {
            if (!string.IsNullOrEmpty(args.Data))
            {
                Debug.Log($"[Python Output] {args.Data}");
            }
        };

        pythonProcess.ErrorDataReceived += (sender, args) =>
        {
            if (!string.IsNullOrEmpty(args.Data))
            {
                Debug.LogError($"[Python Error] {args.Data}");
            }
        };

        try
        {
            Debug.Log($"스크립트 경로: {scriptPath}");
            pythonProcess.Start();
            pythonProcess.BeginOutputReadLine();
            pythonProcess.BeginErrorReadLine();
            Debug.Log("Python 스크립트가 실행되었습니다.");
        }
        catch (System.Exception ex)
        {
            Debug.LogError($"Python 스크립트 실행 오류: {ex.Message}");
        }
    }

    public void StopPython()
    {
        if (pythonProcess == null || pythonProcess.HasExited)
        {
            Debug.LogWarning("실행 중인 Python 프로세스가 없습니다.");
            return;
        }

        try
        {
            if (!pythonProcess.HasExited)
            {
                UnityEngine.Debug.Log("[Video] Python 프로세스 종료 시작");
                pythonProcess.Kill();
                pythonProcess.WaitForExit(2000); // 최대 2초 대기
                UnityEngine.Debug.Log("[Video] Python 프로세스 종료 완료");
            }
            pythonProcess.Close();
        }
        catch (System.Exception ex)
        {
            UnityEngine.Debug.LogError($"Python 스크립트 중지 오류: {ex.Message}");
        }
        finally
        {
            pythonProcess = null;
        }
    }
}