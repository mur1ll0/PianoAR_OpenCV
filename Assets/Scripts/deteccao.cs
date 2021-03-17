using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
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

    Mat TadaptiveMeanBin(Mat gray, int scale, int block_size, int sub_mean, Size morph_kernel, int morph_type, bool negative)
    {
        //Reduzir imagem
        int width = (int)(gray.width() * scale / 100);
        int height = (int)(gray.height() * scale / 100);
        Mat dim = new Mat(height, width, CvType.CV_8UC4);
        Imgproc.resize(gray, dim, dim.size(), 0, 0, Imgproc.INTER_LINEAR);
        //Aplicar Threshold adaptativo
        Mat th = new Mat(height, width, CvType.CV_8UC4);
        Imgproc.adaptiveThreshold(dim, th, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, block_size, sub_mean);
        //Aplicar filtro para reduzir ruidos
        Mat kernel = new Mat( (int)morph_kernel.height, (int)morph_kernel.width, CvType.CV_8U, new Scalar(255));
        //Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, morph_kernel); //Teste kernel ellipse
        Mat morph = new Mat(height, width, CvType.CV_8UC4);
        Imgproc.morphologyEx(th, morph, morph_type, kernel);
        //Voltar ao tamanho original
        width = gray.width();
        height = gray.height();
        Mat defaultDim = new Mat(height, width, CvType.CV_8UC4);
        Imgproc.resize(morph, defaultDim, defaultDim.size(), 0, 0, Imgproc.INTER_LINEAR);
        // Filtro negativo na imagem (Inverter cores)
        Mat result = new Mat(height, width, CvType.CV_8UC4);
        if (negative == true)
        {
            Core.bitwise_not(defaultDim, result);
        }
        //Se não, usa bitwise_and para copiar defaultDim para result
        else Core.bitwise_and(defaultDim, defaultDim, result);
        return result;
    }

    Mat auto_canny(Mat image, float sigma= 0.33f)
    {
        //compute the median of the single channel pixel intensities
        Scalar v = Core.mean(image);
        //apply automatic Canny edge detection using the computed median
        int lower = (int)(Mathf.Max(0, (1 - sigma) * (float)v.val[0]));

        int upper = (int)(Mathf.Min(255, (1 + sigma) * (float)v.val[0]));

        Mat edged = new Mat();
        Imgproc.Canny(image, edged, lower, upper);
        // return the edged image
        return edged;
    }

    Mat removeSombras(Mat img)
    {
        List<Mat> rgb_planes = new List<Mat>();
        Core.split(img, rgb_planes);
        List<Mat> result_norm_planes = new List<Mat>();
        foreach (Mat plane in rgb_planes)
        {
            Mat kernel = new Mat(7, 7, CvType.CV_8U, new Scalar(255));
            Mat dilated_img = new Mat();
            Imgproc.dilate(plane, dilated_img, kernel);
            Mat bg_img = new Mat();
            Imgproc.medianBlur(dilated_img, bg_img, 21);
            Mat white = new Mat(bg_img.height(), bg_img.width(), CvType.CV_8UC4, new Scalar(255));
            Mat diff_img = new Mat();
            Core.absdiff(plane, bg_img, diff_img);
            Core.subtract(white, diff_img, diff_img);
            Mat norm_img = new Mat();
            Core.normalize(diff_img, norm_img, 0, 255, Core.NORM_MINMAX, CvType.CV_8UC1);
            result_norm_planes.Add(norm_img);
        }
        Mat result = new Mat();
        Core.merge(result_norm_planes, result);
        return result;
    }


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
        quad.GetComponent<Renderer>().material.mainTexture = outputTexture;

    }

    // Update is called once per frame
    void Update()
    {
        
    }

    // OnPostRender é chamado depois da câmera renderizar o quadro 
    void OnPostRender()
    {
        //---------------------------------------------------
        //Ler frame da câmera e converter para MAT do OpenCV
        //---------------------------------------------------
        UnityEngine.Rect rect = new UnityEngine.Rect(0, 0, cameraTexture.width, cameraTexture.height);
        cameraTexture.ReadPixels(rect, 0, 0, true);
        Utils.texture2DToMat(cameraTexture, cameraMat, false);
        
        //Converter em escala de cinza
        Mat gray = new Mat(Screen.height, Screen.width, CvType.CV_8UC4);
        Imgproc.cvtColor(cameraMat, gray, Imgproc.COLOR_RGB2GRAY);

        //Calcular area do frame
        double frame_area = Screen.height * Screen.width;

        //Aplicar Threshold
        Mat brancas = TadaptiveMeanBin(gray, 50, 11, 6, new Size(3, 3), 3, false);

        //Teste Canny
        //brancas = auto_canny(gray);

        //Teste remover sombras
        //brancas = removeSombras(gray);

        //Threshold binário
        Mat kernel = new Mat(5, 5, CvType.CV_8U, new Scalar(255));
        Mat blur = new Mat();
        Imgproc.GaussianBlur(gray, blur, new Size(9, 9), 3);
        Imgproc.threshold(blur, brancas, 80, 255, Imgproc.THRESH_BINARY_INV);
        //Imgproc.erode(brancas, brancas, kernel, new Point(), 1);
        //Imgproc.dilate(brancas, brancas, kernel, new Point(), 1);
        

        // Primeira parte: Encontrar ROI das teclas, separando do resto do piano/teclado
        Mat img_sobelx = new Mat(Screen.height, Screen.width, CvType.CV_8U);
        Imgproc.Sobel(brancas, img_sobelx, CvType.CV_8U, 1, 0, 21);
        Mat img_sobely = new Mat(Screen.height, Screen.width, CvType.CV_8U);
        Imgproc.Sobel(brancas, img_sobely, CvType.CV_8U, 0, 1, 21);
        Mat bordas = new Mat();
        Core.add(img_sobelx, img_sobely, bordas);
        

        //Declarar variavel que será o ROI da área das teclas (Mat todo preto)
        Mat keysROI = new Mat(Screen.height, Screen.width, CvType.CV_8U, Scalar.all(0));

        /*
        List<MatOfPoint> srcContours = new List<MatOfPoint>();
        Mat srcHierarchy = new Mat();
        /// Find srcContours
        Imgproc.findContours(bordas, srcContours, srcHierarchy, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_NONE);

        Debug.Log("srcContours.Count " + srcContours.Count);

        for (int i = 0; i < srcContours.Count; i++)
        {
            Imgproc.drawContours(cameraMat, srcContours, i, new Scalar(255, 0, 0), 2, 8, srcHierarchy, 0, new Point());
        }

        for (int i = 0; i < srcContours.Count; i++)
        {
            double returnVal = Imgproc.matchShapes(srcContours[1], srcContours[i], Imgproc.CV_CONTOURS_MATCH_I1, 0);
            Debug.Log("returnVal " + i + " " + returnVal);

            Point point = new Point();
            float[] radius = new float[1];
            Imgproc.minEnclosingCircle(new MatOfPoint2f(srcContours[i].toArray()), point, radius);
            Debug.Log("point.ToString() " + point.ToString());
            Debug.Log("radius.ToString() " + radius[0]);

            Imgproc.circle(cameraMat, point, 5, new Scalar(0, 0, 255), -1);
            Imgproc.putText(cameraMat, " " + returnVal, point, Imgproc.FONT_HERSHEY_SIMPLEX, 0.4, new Scalar(0, 255, 0), 1, Imgproc.LINE_AA, false);
        }*/

        //Contornos
        Mat mask = new Mat(brancas.rows(), brancas.cols(), CvType.CV_8U, Scalar.all(0));
        List<MatOfPoint> contours = new List<MatOfPoint>();
        Imgproc.findContours(brancas, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_TC89_L1);
        foreach (MatOfPoint c in contours)
        {
            double area = Imgproc.contourArea(c);
            OpenCVForUnity.CoreModule.Rect bRect = Imgproc.boundingRect(c);

            double perc_area = (area * 100 / frame_area);

            /*/Se encontrar área que corresponde a mais de 8% do frame original, marcar como ROI
            if (perc_area > 8 && perc_area < 40)
            {
                //Imgproc.rectangle(mask, bRect, new Scalar(0, 0, 255), 5, Imgproc.FILLED);
                List<MatOfPoint> boxContours = new List<MatOfPoint>();
                boxContours.Add(new MatOfPoint(c));
                Imgproc.drawContours(mask, boxContours, 0, new Scalar(0, 0, 255), 2);

                cameraMat.copyTo(keysROI, mask);
                Debug.Log("Perc_area: " + perc_area.ToString());
                break;
            }
            //Imgproc.rectangle(cameraMat, bRect, new Scalar(255, 0, 0), 5, Imgproc.LINE_AA);
            */
            List<MatOfPoint> boxContours = new List<MatOfPoint>();
            boxContours.Add(new MatOfPoint(c));
            Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(0, 0, 255), 2);
        }


        //Imgproc.cvtColor(brancas, cameraMat, Imgproc.COLOR_GRAY2RGB);
        //Imgproc.putText(cameraMat, "Tamanho do FRAME: " + cameraTexture.width + "x" + cameraTexture.height, new Point(5, cameraTexture.height - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, new Scalar(255, 0, 0, 255));

        //-------------------------------------------------------
        //Converter MAT do OpenCV para textura mapeada na câmera
        //------------------------------------------------------
        Utils.matToTexture2D(cameraMat, outputTexture, colors, false);
    }
}
