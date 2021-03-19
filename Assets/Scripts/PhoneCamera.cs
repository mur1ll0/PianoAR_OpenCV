using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgprocModule;
using OpenCVForUnity.UnityUtils;

public class PhoneCamera : MonoBehaviour
{

	private bool camAvailable;
	private WebCamTexture cameraTexture;
	private Texture defaultBackground;

    Mat cameraMat;
    Mat BlackKeyMat;
    Texture2D outputTexture;
    Texture2D BlackKeyTexture;
    Color32[] colors;

    public RawImage background;
	public AspectRatioFitter fit;
	public bool frontFacing;
    public Texture BlackKeyContour;

    // Use this for initialization
    void Start()
	{
		defaultBackground = background.texture;
		WebCamDevice[] devices = WebCamTexture.devices;

		if (devices.Length == 0)
			return;

        Debug.Log("Number of web cams connected: " + devices.Length);

        /*for (int i = 0; i < devices.Length; i++)
		{
			var curr = devices[i];

			if (curr.isFrontFacing == frontFacing)
			{
				cameraTexture = new WebCamTexture(curr.name, Screen.width, Screen.height);
				break;
			}
		}*/

        var curr = devices[0];
        cameraTexture = new WebCamTexture(curr.name, Screen.width, Screen.height);

        if (cameraTexture == null)
            return;

        Debug.Log("Device connected: " + cameraTexture.deviceName);

        cameraTexture.Play(); // Start the camera
        background.texture = cameraTexture; // Set the texture

        cameraMat = new Mat(cameraTexture.height, cameraTexture.width, CvType.CV_8UC4);
        outputTexture = new Texture2D(cameraTexture.width, cameraTexture.height, TextureFormat.ARGB32, false);
        background.texture = outputTexture; // Set the texture

        //Setar tecla preta texture2D pra mat
        BlackKeyTexture = new Texture2D(BlackKeyContour.width,BlackKeyContour.height);
        BlackKeyTexture = BlackKeyContour as Texture2D;
        BlackKeyMat = new Mat(BlackKeyTexture.height, BlackKeyTexture.width, CvType.CV_8UC4);
        Utils.texture2DToMat(BlackKeyTexture, BlackKeyMat, false);

        camAvailable = true; // Set the camAvailable for future purposes.
	}

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
        Mat kernel = new Mat((int)morph_kernel.height, (int)morph_kernel.width, CvType.CV_8U, new Scalar(255));
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

    Mat auto_canny(Mat image, float sigma = 0.33f)
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

    // Update is called once per frame
    void Update()
	{
		if (!camAvailable)
			return;

		float ratio = (float)cameraTexture.width / (float)cameraTexture.height;
		fit.aspectRatio = ratio; // Set the aspect ratio

		float scaleY = cameraTexture.videoVerticallyMirrored ? -1f : 1f; // Find if the camera is mirrored or not
		background.rectTransform.localScale = new Vector3(1f, scaleY, 1f); // Swap the mirrored camera

		int orient = -cameraTexture.videoRotationAngle;
		background.rectTransform.localEulerAngles = new Vector3(0, 0, orient);


        //---------------------------------------------------
        //Ler frame da câmera e converter para MAT do OpenCV
        //---------------------------------------------------
        //outputTexture.SetPixels(cameraTexture.GetPixels());
        outputTexture.SetPixels32(cameraTexture.GetPixels32());
        outputTexture.Apply();
        //UnityEngine.Rect rect = new UnityEngine.Rect(0, 0, cameraTexture.width, cameraTexture.height);
        //outputTexture.ReadPixels(rect, 0, 0, true);
        Utils.texture2DToMat(outputTexture, cameraMat, false);

        //Converter em escala de cinza
        Mat gray = new Mat(Screen.height, Screen.width, CvType.CV_8UC4);
        Imgproc.cvtColor(cameraMat, gray, Imgproc.COLOR_RGB2GRAY);

        //Calcular area do frame
        double frame_area = Screen.height * Screen.width;

        //Threshold binário
        Mat brancas = new Mat();
        Mat kernel = new Mat(5, 5, CvType.CV_8U, new Scalar(255));
        Mat blur = new Mat();
        Imgproc.GaussianBlur(gray, blur, new Size(9, 9), 3);
        Imgproc.threshold(blur, brancas, 80, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.erode(brancas, brancas, kernel, new Point(), 1);
        //Imgproc.dilate(brancas, brancas, kernel, new Point(), 1);


        // Primeira parte: Encontrar ROI das teclas, separando do resto do piano/teclado
        Mat img_sobelx = new Mat(Screen.height, Screen.width, CvType.CV_8U);
        Imgproc.Sobel(brancas, img_sobelx, CvType.CV_8U, 1, 0, 21);
        Mat img_sobely = new Mat(Screen.height, Screen.width, CvType.CV_8U);
        Imgproc.Sobel(brancas, img_sobely, CvType.CV_8U, 0, 1, 21);
        Mat bordas = new Mat();
        Core.add(img_sobelx, img_sobely, bordas);


        //Declarar variavel que será o ROI da área das teclas (Mat todo preto)
        Mat keysROI = new Mat(Screen.height, Screen.width, CvType.CV_8UC4, Scalar.all(0));
        Imgproc.threshold(gray, keysROI, 0, 0, Imgproc.THRESH_BINARY);

        //Contorno da tecla preta para usar como match
        Mat BlackKeyMatGray = new Mat();
        Imgproc.cvtColor(BlackKeyMat, BlackKeyMatGray, Imgproc.COLOR_RGB2GRAY);
        List<MatOfPoint> contourPreta = new List<MatOfPoint>();
        Imgproc.findContours(BlackKeyMatGray, contourPreta, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_TC89_L1);
        MatOfPoint cPreta = contourPreta[0];
        OpenCVForUnity.CoreModule.Rect bRectBlackKey = Imgproc.boundingRect(cPreta);
        //double aspect_ratio_blackKey = cPreta.height() / cPreta.width();
        double aspect_ratio_blackKey = bRectBlackKey.height / bRectBlackKey.width;
        //Debug.Log("-----------------");
        //Debug.Log("BlackKey H:" + bRectBlackKey.height.ToString() + " W:" + bRectBlackKey.width.ToString());
        //Debug.Log("BlackKey AspectRatio:" + aspect_ratio_blackKey.ToString("0.###"));
        //Debug.Log("-----------------");

        ////Template Matching with Multiple Objects
        //Mat matches = new Mat();
        //Imgproc.threshold(gray, matches, 0, 0, Imgproc.THRESH_BINARY);
        //Imgproc.matchTemplate(brancas, BlackKeyMat, matches, Imgproc.TM_CCOEFF_NORMED);
        //Imgproc.threshold(matches, matches, 0.1, 1, Imgproc.THRESH_TOZERO);
        ////Imgproc.threshold(cameraMat, matches, 0, 0, Imgproc.THRESH_BINARY);
        ////Imgproc.matchTemplate(cameraMat, BlackKeyMat, matches, Imgproc.TM_CCOEFF_NORMED);
        ////Imgproc.threshold(matches, matches, 0.1, 1, Imgproc.THRESH_TOZERO);
        //double threshold = 0.95;
        //double maxval;
        //Mat dst;
        //while (true)
        //{
        //    Core.MinMaxLocResult maxr = Core.minMaxLoc(matches);
        //    Point maxp = maxr.maxLoc;
        //    maxval = maxr.maxVal;
        //    Point maxop = new Point(maxp.x + BlackKeyMat.width(), maxp.y + BlackKeyMat.height());
        //    dst = brancas.clone();
        //    if (maxval >= threshold)
        //    {

        //        Imgproc.rectangle(cameraMat, maxp, new Point(maxp.x + BlackKeyMat.cols(),
        //                maxp.y + BlackKeyMat.rows()), new Scalar(0, 255, 0), 5);
        //        Imgproc.rectangle(matches, maxp, new Point(maxp.x + BlackKeyMat.cols(),
        //                maxp.y + BlackKeyMat.rows()), new Scalar(0, 255, 0), -1);
        //    }
        //    else
        //    {
        //        break;
        //    }
        //}

        //Contador de teclas detectadas
        int detected_count = 0;

        //Contornos
        Mat mask = new Mat(brancas.rows(), brancas.cols(), CvType.CV_8U, Scalar.all(0));
        List<MatOfPoint> contours = new List<MatOfPoint>();
        Imgproc.findContours(brancas, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_TC89_L1);
        foreach (MatOfPoint c in contours)
        {
            double area = Imgproc.contourArea(c);
            OpenCVForUnity.CoreModule.Rect bRect = Imgproc.boundingRect(c);

            double perc_area = (area * 100 / frame_area);

            //Eliminar ruidos nos contornos
            if (perc_area < 0.1)
            {
                //continue;    
            }

            ////Se encontrar área que corresponde a mais de 8 % do frame original, marcar como ROI
            //if (perc_area > 8 && perc_area < 40)
            //{
            //    Imgproc.rectangle(mask, bRect, new Scalar(0, 0, 255), 5, Imgproc.FILLED);
            //    List<MatOfPoint> boxContours = new List<MatOfPoint>();
            //    boxContours.Add(new MatOfPoint(c));
            //    Imgproc.drawContours(mask, boxContours, 0, new Scalar(0, 0, 255), 2);

            //    cameraMat.copyTo(keysROI, mask);
            //    Debug.Log("Perc_area: " + perc_area.ToString());
            //    break;
            //}
            //Imgproc.rectangle(cameraMat, bRect, new Scalar(255, 0, 0), 5, Imgproc.LINE_AA);


            ////Mostrar contorno
            //List<MatOfPoint> boxContours = new List<MatOfPoint>();
            //boxContours.Add(new MatOfPoint(c));
            //Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(0, 0, 255), 2);



            /*/Fazer match dos contornos para achar somente as teclas pretas
            double valor_match = Imgproc.matchShapes(cPreta, c, Imgproc.CONTOURS_MATCH_I1, 0);
            
            if (valor_match < 0.1)
            {
                //Se tiver mais de 70% de chance de ser um tecla, pinta no keysROI
                //Imgproc.rectangle(mask, bRect, new Scalar(0, 0, 255), 5, Imgproc.FILLED);
                List<MatOfPoint> boxContours = new List<MatOfPoint>();
                boxContours.Add(new MatOfPoint(c));
                //Imgproc.drawContours(mask, boxContours, 0, new Scalar(255), 2);
                cameraMat.copyTo(keysROI, mask);

                Imgproc.putText(cameraMat, " " + valor_match.ToString("0.##"), new Point(bRect.x, bRect.y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 0, 0), 1, Imgproc.LINE_AA, true);
                Imgproc.rectangle(cameraMat, bRect, new Scalar(0,0,255), 1, Imgproc.LINE_AA);
            }

            List<MatOfPoint> boxContours2 = new List<MatOfPoint>();
            boxContours2.Add(cPreta);
            Imgproc.drawContours(cameraMat, boxContours2, 0, new Scalar(255,0,0), 2);
            */

            ////Match por aspect ratio
            ////double aspect_ratio_contour = c.height() / c.width();
            //double aspect_ratio_contour = bRect.height / bRect.width;
            //double perc_match_ratio = aspect_ratio_contour * 100 / aspect_ratio_blackKey;
            //if ((perc_match_ratio > 10) && (perc_match_ratio < 190)){
            //    List<MatOfPoint> boxContours = new List<MatOfPoint>();
            //    boxContours.Add(c);
            //    Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(255, 0, 0), 2);

            //    Debug.Log("DETECTED: H:"+bRect.height.ToString()+" W:"+bRect.width.ToString()+" AR:"+aspect_ratio_contour.ToString("0.###")+" MatchAR:"+perc_match_ratio.ToString("0.#####"));
            //}
            //else
            //{
            //    Debug.Log("NOT: H:" + bRect.height.ToString() + " W:" + bRect.width.ToString() + " AR:" + aspect_ratio_contour.ToString("0.###") + " MatchAR:" + perc_match_ratio.ToString("0.#####"));
            //}

            //Desenhar apenas os contornos com Aspect Ratio entre 1 e 15
            double aspect_ratio_contour = bRect.height / bRect.width;

            //if (aspect_ratio_contour>5 && aspect_ratio_contour < 10)
            //{
            //    List<MatOfPoint> boxContours = new List<MatOfPoint>();
            //    boxContours.Add(c);
            //    Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(255, 0, 0), 2);
            //}

            //Aproximação do contorno
            MatOfPoint2f cPoly = new MatOfPoint2f();
            MatOfPoint2f c2f = new MatOfPoint2f();
            c.convertTo(c2f, CvType.CV_32FC2);
            double epsilon = Imgproc.arcLength(c2f, true) * 0.02;
            Imgproc.approxPolyDP(new MatOfPoint2f(c.toArray()), cPoly, 3, true);
            //AR da aproximação
            double aspect_ratio_approx = cPoly.height() / cPoly.width();

            //if (aspect_ratio_approx > 1 && aspect_ratio_approx < 15)
            if (aspect_ratio_contour > 1 && aspect_ratio_contour < 15)
            {
                Point center = new Point();
                float[] radius = new float[1];
                Imgproc.minEnclosingCircle(cPoly, center, radius);
                //Imgproc.circle(cameraMat, center, (int)radius[0], new Scalar(0, 0, 255), 2);
                Imgproc.putText(cameraMat, " "+ aspect_ratio_approx.ToString("0.##"), center, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 0, 0), 1, Imgproc.LINE_AA, true);

                MatOfPoint approxContour = new MatOfPoint();
                cPoly.convertTo(approxContour, CvType.CV_32S);
                List<MatOfPoint> boxContours = new List<MatOfPoint>();
                boxContours.Add(approxContour);
                Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(255, 0, 0), 2);

                detected_count++;
            }

        }

        Debug.Log("Teclas detectadas: "+detected_count.ToString());

        //Imgproc.cvtColor(brancas, cameraMat, Imgproc.COLOR_GRAY2RGB);
        //Imgproc.putText(cameraMat, "Tamanho do FRAME: " + cameraTexture.width + "x" + cameraTexture.height, new Point(5, cameraTexture.height - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, new Scalar(255, 0, 0, 255));

        //-------------------------------------------------------
        //Converter MAT do OpenCV para textura mapeada na câmera
        //------------------------------------------------------
        Utils.matToTexture2D(cameraMat, outputTexture, false);

    }
}
