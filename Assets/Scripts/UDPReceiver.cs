using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Text;
using System.Net.Sockets;
using System.Threading;
using System.Net;
//using static vit;
using System.Runtime.CompilerServices;
using System.Diagnostics;

public class UDPReceiver : MonoBehaviour
{
    Thread keypoints_Thread;
    UdpClient keypoints_client;
    public int keypoints_port = 5252;
    public bool keypoints_startReceiving = true;
    public string keypoints_data;
    public Keypoints keypoints;
    private object keypointsLock = new object();
    public bool keypoints_dataReceived = false;

    Thread image_Thread;
    UdpClient image_client;
    public int image_port = 5253;
    public bool image_startReceiving = true;
    public bool image_dataReceived = false;
    public byte[] image_data;

    // Start is called before the first frame update
    public void Start()
    {

        keypoints_Thread = new Thread(new ThreadStart(ReceiveKeypoints));
        keypoints_Thread.IsBackground= true;
        keypoints_Thread.Start();

        //image_Thread = new Thread(new ThreadStart(ReceiveImage));
        //image_Thread.IsBackground = true;
        //image_Thread.Start();
    }

    private void ReceiveKeypoints()
    {
        keypoints_client = new UdpClient(keypoints_port);
        while (keypoints_startReceiving)
        {
            IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
            byte[] dataByte = keypoints_client.Receive(ref anyIP);
            keypoints_data = Encoding.UTF8.GetString(dataByte);
            Keypoints receivedKeypoints = JsonUtility.FromJson<Keypoints>(keypoints_data);
            if (keypoints != null)
            {
                lock (keypointsLock)
                {
                    keypoints = receivedKeypoints;
                }
                keypoints_dataReceived = true;
            }
        }
    }
    //private void ReceiveImage()
    //{
    //    image_client = new UdpClient(image_port);
    //    while (image_startReceiving)
    //    {
    //        IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
    //        byte[] dataByte = image_client.Receive(ref anyIP);
    //        image_data = dataByte;
    //        image_dataReceived = true;
    //    }
    //}
}
