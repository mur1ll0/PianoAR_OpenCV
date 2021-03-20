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
    public double areaExcludeValue;

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


        //Teclado
        if (Input.GetKeyDown(KeyCode.KeypadPlus))
        {
            areaExcludeValue += 0.01;
        }
        else if (Input.GetKeyDown(KeyCode.KeypadMinus))
        {
            areaExcludeValue -= 0.01;
        }

        //---------------------------------------------------
        //Ler frame da câmera e converter para MAT do OpenCV
        //---------------------------------------------------
        outputTexture.SetPixels32(cameraTexture.GetPixels32());
        outputTexture.Apply();
        Utils.texture2DToMat(outputTexture, cameraMat, false);

        //Converter em escala de cinza
        Mat gray = new Mat(Screen.height, Screen.width, CvType.CV_8UC4);
        Imgproc.cvtColor(cameraMat, gray, Imgproc.COLOR_RGB2GRAY);

        //Calcular area do frame
        double frame_area = Screen.height * Screen.width;

        //Threshold binário
        Mat th = new Mat();
        Mat kernel = new Mat(5, 5, CvType.CV_8U, new Scalar(255));
        Mat blur = new Mat();
        Imgproc.GaussianBlur(gray, blur, new Size(9, 9), 3);
        Imgproc.threshold(blur, th, 80, 255, Imgproc.THRESH_BINARY_INV);
        Imgproc.erode(th, th, kernel, new Point(), 1);
        //Imgproc.dilate(brancas, brancas, kernel, new Point(), 1);

        //--------------------------------
        // DECLARAÇÕES
        //--------------------------------
        //Declarar variavel que será o ROI da área das teclas (Mat todo preto)
        Mat keysROI = new Mat(Screen.height, Screen.width, CvType.CV_8UC4, Scalar.all(0));
        Imgproc.threshold(gray, keysROI, 0, 0, Imgproc.THRESH_BINARY);
        //Contador de contornos detectados
        int detectedCount = 0;
        //Lista de contornos detectados
        List<MatOfPoint> detectedContours = new List<MatOfPoint>();
        List<MatOfPoint2f> detectedApprox = new List<MatOfPoint2f>();
        List<OpenCVForUnity.CoreModule.Rect> detectedRect = new List<OpenCVForUnity.CoreModule.Rect>();
        //Limites da área detectada
        int xMax = 0;
        int xMin = cameraMat.width();
        int yMax = 0;
        int yMin = cameraMat.height();
        //Soma dos X e Y, para dividir depois por detected_count e achar a média do x e y
        double xMaxSum = 0;
        double xMinSum = 0;
        double yMaxSum = 0;
        double yMinSum = 0;
        //Vertices usados para retângulo rotacionado
        Point[] vertices = new Point[4];
        //Lista de contornos usada para drawContour
        List<MatOfPoint> boxContours = new List<MatOfPoint>();

        //Contornos
        //Mat mask = new Mat(brancas.rows(), brancas.cols(), CvType.CV_8U, Scalar.all(0));
        List<MatOfPoint> contours = new List<MatOfPoint>();
        Imgproc.findContours(th, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_TC89_L1);
        foreach (MatOfPoint c in contours)
        {
            //Contorno convertido em MatOfPoint2f
            MatOfPoint2f c2f = new MatOfPoint2f();
            c.convertTo(c2f, CvType.CV_32FC2);

            //----------------------
            // Métricas do contorno
            //----------------------
            //Área
            double cArea = Imgproc.contourArea(c);
            //Perímetro
            double cPerimeter = Imgproc.arcLength(c2f, true);
            //Aspect Ratio (H/W)
            double cAR = c.height() / c.width();

            //-------------------------
            // Aproximações do contorno
            //-------------------------
            //Retângulo máximo
            OpenCVForUnity.CoreModule.Rect bRect = Imgproc.boundingRect(c);
            //Retângulo rotacionado
            RotatedRect rRect = Imgproc.minAreaRect(c2f);
            //Círculo
            Point cCenter = new Point();
            float[] cRadius = new float[1];
            Imgproc.minEnclosingCircle(c2f, cCenter, cRadius);
            //Aproximação ApproxPolyDP
            MatOfPoint2f cPoly = new MatOfPoint2f();
            Imgproc.approxPolyDP(c2f, cPoly, 3, true);


            //Eliminar áreas muito pequenas (ruídos) nos contornos
            double perc_area = (cArea * 100 / frame_area);
            //if (perc_area < 0.05)
            if (perc_area < areaExcludeValue)
            {
                continue;    
            }


            //---------------------------
            // Selecionar contornos úteis
            //---------------------------

            //Aspect Ratio do retângulo máximo (H/W)
            double rAR = bRect.height / bRect.width;
            //Aspect Ratio do retângulo rotacionado (H/W)
            double rrAR = rRect.size.height / rRect.size.width;
            if ( Mathf.Abs((float)rRect.angle) > 45) rrAR = rRect.size.width / rRect.size.height;

            //Selecionar contornos com Aspect Ratio do Retângulo Rotacionado entre 3 e 12
            if (rrAR > 3 && rrAR < 12)
            //if (cAR > 1 && cAR < 15)
            {
                //Salvar contornos
                detectedContours.Add(c);
                detectedApprox.Add(cPoly);
                detectedRect.Add(bRect);

                //Somar x e y
                xMinSum += bRect.x;
                xMaxSum += bRect.x + bRect.width;
                yMinSum += bRect.y;
                yMaxSum += bRect.y + bRect.height;

                //Calcular máximos e mínimos
                if (bRect.x < xMin) xMin = bRect.x;
                if (bRect.x + bRect.width > xMax) xMax = bRect.x + bRect.width;
                if (bRect.y < yMin) yMin = bRect.y;
                if (bRect.y + bRect.height > yMax) yMax = bRect.y + bRect.height;

                //Incrementar contador de selecionados/detectados
                detectedCount++;

                //Desenhar aproximação ApproxPolyDP
                //MatOfPoint approxContour = new MatOfPoint();
                //cPoly.convertTo(approxContour, CvType.CV_32S);
                //List<MatOfPoint> boxContours = new List<MatOfPoint>();
                //boxContours.Add(approxContour);
                //Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(255, 0, 0), 2);

                //Desenhar Retângulo rotacionado
                rRect.points(vertices);
                boxContours.Clear();
                boxContours.Add(new MatOfPoint(vertices));
                Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(255, 0, 0), 1);
            }


            //Desenhar circulo
            //Imgproc.circle(cameraMat, cCenter, (int)cRadius[0], new Scalar(0, 0, 255), 2);

            //Desenhar retângulo máximo
            //Imgproc.rectangle(cameraMat, bRect, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);

            //Desenhar Retângulo rotacionado
            //rRect.points(vertices);
            //boxContours.Clear();
            //boxContours.Add(new MatOfPoint(vertices));
            //Imgproc.drawContours(cameraMat, boxContours, 0, new Scalar(255, 0, 0), 1);

            //Imprimir AspectRatio
            Imgproc.putText(cameraMat, " " + rrAR.ToString("0.##"), cCenter, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 0, 0), 1, Imgproc.LINE_AA, true);

        }

        Debug.Log("Teclas detectadas: "+detectedCount.ToString());

        //-------------------------------
        // Definir área das teclas pretas
        //-------------------------------
        //Percorrer os contornos selecionados e definir limites da área
        foreach (OpenCVForUnity.CoreModule.Rect dRect in detectedRect)
        {
            //Média de X e Y
            double xMinMedia = xMinSum / detectedCount;
            double xMaxMedia = xMaxSum / detectedCount;
            double yMinMedia = yMinSum / detectedCount;
            double yMaxMedia = yMaxSum / detectedCount;

            //Descartar contornos cujo x/y seja 10% maior ou menor do que a média
            if (true)
            {

            }
        }

        Imgproc.putText(cameraMat, " " + areaExcludeValue.ToString("0.##"), new Point(150, 15), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(255, 0, 0), 1, Imgproc.LINE_AA, true);

        //Imgproc.cvtColor(th, cameraMat, Imgproc.COLOR_GRAY2RGB);
        //Imgproc.putText(cameraMat, "Tamanho do FRAME: " + cameraTexture.width + "x" + cameraTexture.height, new Point(5, cameraTexture.height - 5), Imgproc.FONT_HERSHEY_SIMPLEX, 2.0, new Scalar(255, 0, 0, 255));

        //-------------------------------------------------------
        //Converter MAT do OpenCV para textura mapeada na câmera
        //------------------------------------------------------
        Utils.matToTexture2D(cameraMat, outputTexture, false);

    }
}
