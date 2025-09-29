using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class JointData : MonoBehaviour
{
    [Header("프레임 설정")]
    public float targetFPS = 15.0f;
    
    [Header("객체 데이터")]
    [HideInInspector]
    public ObjectData objectData_0;
    [HideInInspector]
    public ObjectData objectData_1;
    [HideInInspector]
    public ObjectData objectData_2;
    
    private JointDataExtractor extractor;
    
    void Start()
    {
        extractor = GetComponent<JointDataExtractor>();
        extractor.SetTargetFPS(targetFPS);
        extractor.enabled = true;
    }

    void Update()
    {
        
    }
    
    // JointDataExtractor에서 새로운 데이터가 처리되었을 때 호출되는 메서드
    public void OnObjectDataUpdated()
    {
        // JointDataExtractor에서 처리된 객체 데이터 가져오기
        if (extractor != null)
        {
            objectData_0 = extractor.GetObjectData(0);
            objectData_1 = extractor.GetObjectData(1);
            objectData_2 = extractor.GetObjectData(2);
            
            int objectCount = 0;
            
            if (objectData_0 != null)
            {
                objectCount++;
                ProcessObjectData(objectData_0);
            }
            
            if (objectData_1 != null)
            {
                objectCount++;
                ProcessObjectData(objectData_1);
            }
            
            if (objectData_2 != null)
            {
                objectCount++;
                ProcessObjectData(objectData_2);
            }
            UnityEngine.Debug.Log($"[DEBUG] JointData - 총 {objectCount}개 객체 처리 완료");
        }
    }
    
    private void ProcessObjectData(ObjectData objData)
    {
        var positions = objData.GetPositionsAsDictionary();
    }
}
