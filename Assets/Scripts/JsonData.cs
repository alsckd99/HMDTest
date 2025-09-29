using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

[System.Serializable]
public class Keypoints
{
    public ObjectData[] objects;


}
[System.Serializable]
public class ObjectData
{
    // 객체 ID
    public int id;
    
    // 얼굴 부분
    public float[] nose;
    public float[] left_eye;
    public float[] right_eye;
    public float[] left_ear;
    public float[] right_ear;
    
    // 상체 부분
    public float[] left_shoulder;
    public float[] right_shoulder;
    public float[] left_elbow;
    public float[] right_elbow;
    public float[] left_wrist;
    public float[] right_wrist;
    
    // 하체 부분
    public float[] left_hip;
    public float[] right_hip;
    public float[] left_knee;
    public float[] right_knee;
    public float[] left_ankle;
    public float[] right_ankle;

    public List<float[]> GetAllPositions()
    {
        return new List<float[]>
        {
            nose, left_eye, right_eye, left_ear, right_ear,
            left_shoulder, right_shoulder, left_elbow, right_elbow, 
            left_wrist, right_wrist, left_hip, right_hip, 
            left_knee, right_knee, left_ankle, right_ankle
        };
    }

    public Dictionary<string, float[]> GetPositionsAsDictionary()
    {
        return new Dictionary<string, float[]>
        {
            { "nose", nose },
            { "left_eye", left_eye },
            { "right_eye", right_eye },
            { "left_ear", left_ear },
            { "right_ear", right_ear },
            { "left_shoulder", left_shoulder },
            { "right_shoulder", right_shoulder },
            { "left_elbow", left_elbow },
            { "right_elbow", right_elbow },
            { "left_wrist", left_wrist },
            { "right_wrist", right_wrist },
            { "left_hip", left_hip },
            { "right_hip", right_hip },
            { "left_knee", left_knee },
            { "right_knee", right_knee },
            { "left_ankle", left_ankle },
            { "right_ankle", right_ankle }
        };
    }

    // 각 객체의 정보를 출력하는 메서드 (ID 포함)
    public void PrintObjectData()
    {
        Debug.Log($"=== Object ID {id} 데이터 ===");
        var positions = GetPositionsAsDictionary();
        
        foreach (var kvp in positions)
        {
            if (kvp.Value != null && kvp.Value.Length >= 3)
            {
                float x = kvp.Value[0];
                float y = kvp.Value[1];
                float confidence = kvp.Value[2];
                
                // 모든 데이터를 출력 (유효성 검사 없음)
                Debug.Log($"{kvp.Key}: X={x:F2}, Y={y:F2}, Confidence={confidence:F3}");
            }
            else
            {
                Debug.Log($"{kvp.Key}: 데이터 없음");
            }
        }
    }
}


