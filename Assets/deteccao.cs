using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Vuforia;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;

public class deteccao : MonoBehaviour
{

    //Declarações
    public GameObject quad;
    public Camera mainCamera;
    Mat cameraMat;
    Texture2D cameraTexture;
    Texture2D outputTexture;
    Color32[] colors;
    Mat mSepiaKernel;

    // Start is called before the first frame update
    void Start()
    {
        cameraMat = new Mat(Screen.height, Screen.width, CvType.CV_8UC4);
        cameraTexture = new Texture2D(cameraMat.cols(), cameraMat.rows(), TextureFormat.ARGB32, false);
        outputTexture = new Texture2D(cameraMat.cols(), cameraMat.rows(), TextureFormat.ARGB32, false);

        colors = new Color32[outputTexture.width * outputTexture.height];
        mainCamera.orthographicSize = cameraTexture.height / 2;

        //Mapear outputTexture para exibir na câmera
        quad.transform.localScale = new Vector3(cameraTexture.width, cameraTexture.height, quad.transform.localScale.z);
        quad.GetComponent<Renderer> ().material.mainTexture = outputTexture;


        // sepia
        mSepiaKernel = new Mat(4, 4, CvType.CV_32F);
        mSepiaKernel.put(0, 0, /* R */0.189f, 0.769f, 0.393f, 0f);
        mSepiaKernel.put(1, 0, /* G */0.168f, 0.686f, 0.349f, 0f);
        mSepiaKernel.put(2, 0, /* B */0.131f, 0.534f, 0.272f, 0f);
        mSepiaKernel.put(3, 0, /* A */0.000f, 0.000f, 0.000f, 1f);
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // OnPostRender é chamado quando a câmera termina de renderizar o quadro 
    void OnPostRender()
    {
        //---------------------------------------------------
        //Ler frame da câmera e converter para MAT do OpenCV
        //---------------------------------------------------
        UnityEngine.Rect rect = new UnityEngine.Rect(0, 0, cameraTexture.width, cameraTexture.height);
        cameraTexture.ReadPixels(rect, 0, 0, true);
        Utils.texture2DToMat(cameraTexture, cameraMat);

        //Converter em escala de cinza
        Mat gray = new Mat(Screen.height, Screen.width, CvType.CV_8UC4); ;
        Imgproc.cvtColor(cameraMat, gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.putText(cameraMat, "CINZOU " + cameraTexture.width + "x" + cameraTexture.height, new Point(5, cameraTexture.height - 5), Imgproc.FONT_HERSHEY_PLAIN, 1.0, new Scalar(255, 0, 0, 255));
        //cameraMat = gray;

        Core.transform(cameraMat, cameraMat, mSepiaKernel);

        //-------------------------------------------------------
        //Converter MAT do OpenCV para textura mapeada na câmera
        //------------------------------------------------------
        Utils.matToTexture2D(cameraMat, outputTexture, colors);
    }
}
