package com.example.tomasz.opencv;

import android.content.Context;
import android.content.ContextWrapper;
import android.graphics.Bitmap;
import android.hardware.camera2.CameraManager;
import android.media.FaceDetector;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Base64;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2, View.OnClickListener {

    private CascadeClassifier cascadeClassifier, cascadeClassifierEye1, cascadeClassifierEye2, cascadeClassifierMouth,cascadeClassifierNose;
    private int absoluteFaceSize;
    private Mat grayscaleImage;
    private int cameraID=1;
    private Button capture, check;
    Mat mRgbaROI;
    private int color=0;
    FeatureDetector detector;
    DescriptorExtractor descriptor;
    DescriptorMatcher matcher;

    Mat descriptors1;
    MatOfKeyPoint keypoints1;
    Mat descriptors2;
    MatOfKeyPoint keypoints2;
    Mat img2,img1;

    ImageView image;

    double faceX,faceY, eye1X,eye1Y,eye2X,eye2Y, mouthX,mouthY, noseX,noseY;
    double [] patternList = new double[8];
    double [] checkList = new double[8];
    String person=null;
    double d1, d2, d3,d4;


    CameraBridgeViewBase javaCameraView;
    private static final String TAG = "MainActivity";
    Mat mRgba, imgCanny, imgGray;
    BaseLoaderCallback mLoaderCallBack =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch(status){
                case BaseLoaderCallback.SUCCESS:{
                   // javaCameraView.enableView();
                    initializeOpenCVDependencies();
                    break;
                }
                default:{
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        javaCameraView = (CameraBridgeViewBase)findViewById(R.id.java_camera_view);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
        javaCameraView.setCameraIndex(cameraID);
        javaCameraView.enableFpsMeter();

        capture = (Button) findViewById(R.id.capture);
        capture.setOnClickListener(this);

        check = (Button) findViewById(R.id.check);
        check.setOnClickListener(this);

        image = (ImageView)findViewById(R.id.imageView);

     /*   detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);;
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
*/

    }



    private void initializeOpenCVDependencies() {

        try {
            // Copy the resource into a temp file so OpenCV can load it
            // stworzyć folder "raw" w C:\Users\Tomasz\AndroidStudioProjects\OpenCV\app\src\main\res\raw
            InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
            //dla twarzy
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // dla oczu
            //lewe
            InputStream is2 = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
            File mcascadeFileEye1 = new File(cascadeDir, "haarcascade_lefteye_2splits.xml");
            FileOutputStream os2 = new FileOutputStream(mcascadeFileEye1);
            byte[] bufferEye1 = new byte[4096];
            int bytesReadEye1;
            while ((bytesReadEye1 = is2.read(bufferEye1)) != -1) {
                os2.write(bufferEye1, 0, bytesReadEye1);
            }
            is2.close();
            os2.close();
            //prawe
            InputStream is3 = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
            File mcascadeFileEye2 = new File(cascadeDir, "haarcascade_righteye_2splits.xml");
            FileOutputStream os3 = new FileOutputStream(mcascadeFileEye2);
            byte[] bufferEye2 = new byte[4096];
            int bytesReadEye2;
            while ((bytesReadEye2 = is3.read(bufferEye2)) != -1) {
                os3.write(bufferEye2, 0, bytesReadEye2);
            }
            is3.close();
            os3.close();

            //dla ust
            InputStream is4 = getResources().openRawResource(R.raw.mouth);
            File mcascadeFileMouth = new File(cascadeDir, "mouth.xml");
            FileOutputStream os4 = new FileOutputStream(mcascadeFileMouth);
            byte[] bufferMouth = new byte[4096];
            int bytesReadMouth;
            while ((bytesReadMouth = is4.read(bufferMouth)) != -1) {
                os4.write(bufferMouth, 0, bytesReadMouth);
            }
            is4.close();
            os4.close();

            //dla nosa
           InputStream is5 = getResources().openRawResource(R.raw.nose);
            File mcascadeFileNose = new File(cascadeDir, "nose.xml");
            FileOutputStream os5 = new FileOutputStream(mcascadeFileNose);
            byte[] bufferNose = new byte[4096];
            int bytesReadNose;
            while ((bytesReadNose = is5.read(bufferNose)) != -1) {
                os5.write(bufferNose, 0, bytesReadNose);
            }
            is5.close();
            os5.close();

            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            cascadeClassifierEye1 = new CascadeClassifier(mcascadeFileEye1.getAbsolutePath());
            cascadeClassifierEye2 = new CascadeClassifier(mcascadeFileEye2.getAbsolutePath());
            cascadeClassifierMouth = new CascadeClassifier(mcascadeFileMouth.getAbsolutePath());
            cascadeClassifierNose = new CascadeClassifier(mcascadeFileNose.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }


        // And we are ready to go
        javaCameraView.enableView();
    }

    @Override
    protected void onPause(){
        super.onPause();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }
    // zdarzenia w momencie wznowienia mainactivity
    @Override
    protected void onResume(){
        super.onResume();
        if(OpenCVLoader.initDebug()){
            Log.i(TAG, "OpenCV loaded");
            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }else{
            Log.i(TAG, "OpenCV not loaded");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0,this,mLoaderCallBack);
        }
    }
    // zdarzenia po zainicjowaniu kamery
    @Override
    public void onCameraViewStarted(int width, int height) {
       //mRgba = new Mat(height,width,CvType.CV_8UC4);
        //imgGray = new Mat(height,width,CvType.CV_8UC1);
        //imgCanny = new Mat(height,width,CvType.CV_8UC1);
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);

             // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.5);
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
    }

    // zdarzenia po pobraniu ramki z kamery
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        //zmiana orientacji wyświetlanego obrazu
        if(cameraID==1) {
            Core.transpose(mRgba, mRgba);
            Core.flip(mRgba, mRgba, -1);
        }else{
            Core.transpose(mRgba, mRgba);
            Core.flip(mRgba, mRgba, 1);
        }
        // przerobienie obrazu rgb na obraz w skali szarości
        if(color==1) {
            Imgproc.cvtColor(mRgba, mRgba, Imgproc.COLOR_RGBA2GRAY);
        }


        //return mRgba;
        return detectElements();
    }

    //wykrwanie elementów twarz i wyrysowywanie prostokątów wokół nich
    public Mat detectElements(){
        MatOfRect faces = new MatOfRect();

        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(mRgba, faces, 1.1, 4, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }
        if(faces.empty()){
            person=null;
        }
        // wyrysowanie prostokąta wokół twarzy
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i <facesArray.length; i++) {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
            //podpisanie prostokąta
            if(person!=null)
            Imgproc.putText(mRgba,person,new Point(facesArray[i].x-5,facesArray[i].y-5),Core.FONT_ITALIC,1,new Scalar(0,255,0,255),4);
            faceX = facesArray[i].x;
            faceY = facesArray[i].y;
           // wybór ROI(twarzy) do dalszego przetwarzania
            mRgbaROI = new Mat(mRgba,facesArray[i]);
            //wykrycie i rysowanie prostokąta dla lewego oka
            MatOfRect eyes1 = new MatOfRect();
            cascadeClassifierEye1.detectMultiScale(mRgbaROI, eyes1, 1.1, 4, 2,
                    new Size(50,50), new Size());
            Rect[] eyesArray1 = eyes1.toArray();
            for(int j=0;j<eyesArray1.length;j++){
                Imgproc.rectangle(mRgbaROI, eyesArray1[i].tl(), eyesArray1[i].br(), new Scalar(255, 0, 0, 255), 3);
                //podpisanie prostokąta
                Imgproc.putText(mRgbaROI,"oko",new Point(eyesArray1[i].x-5,eyesArray1[i].y-5),Core.FONT_ITALIC,1,new Scalar(0,100,0,255),2);
                eye1X = eyesArray1[i].tl().x-eyesArray1[i].br().x;
                eye1Y = eyesArray1[i].tl().y-eyesArray1[i].br().y;
            }
            //wykrycie i rysowanie prostokąta dla prawego oka
            MatOfRect eyes2 = new MatOfRect();
            cascadeClassifierEye2.detectMultiScale(mRgbaROI, eyes2, 1.1, 4, 2,
                    new Size(50,50), new Size());
            Rect[] eyesArray2 = eyes2.toArray();
            for(int j=0;j<eyesArray2.length;j++){
                Imgproc.rectangle(mRgbaROI, eyesArray2[i].tl(), eyesArray2[i].br(), new Scalar(255, 0, 0, 255), 3);
                //podpisanie prostokąta
                Imgproc.putText(mRgbaROI,"oko",new Point(eyesArray2[i].x-5,eyesArray2[i].y-5),Core.FONT_ITALIC,1,new Scalar(0,100,0,255),2);
                eye2X = eyesArray2[i].tl().x-eyesArray2[i].br().x;
                eye2Y = eyesArray2[i].tl().y-eyesArray2[i].br().y;
            }
            //wykrycie i rysowanie prostokąta dla ust
            MatOfRect mouth = new MatOfRect();
            cascadeClassifierMouth.detectMultiScale(mRgbaROI, mouth, 1.1,4, 2,
                    new Size(30,73), new Size());
            Rect[] mouthArray = mouth.toArray();
            for(int j=0;j<mouthArray.length;j++){
                Imgproc.rectangle(mRgbaROI, mouthArray[i].tl(),mouthArray[i].br(), new Scalar(255, 0, 0, 255), 3);
                //podpisanie prostokąta
                Imgproc.putText(mRgbaROI,"usta",new Point(mouthArray[i].x-5,mouthArray[i].y-5),Core.FONT_ITALIC,1,new Scalar(0,100,0,255),2);
                mouthX = mouthArray[i].tl().x-mouthArray[i].br().x;
                mouthY = mouthArray[i].tl().y-mouthArray[i].br().y;
            }
            //wykrycie i rysowanie prostokąta dla nosa
            MatOfRect nose = new MatOfRect();
            cascadeClassifierNose.detectMultiScale(mRgbaROI, nose, 1.1, 4, 2,
                    new Size(40,40), new Size());
            Rect[] noseArray = nose.toArray();
            for(int j=0;j<noseArray.length;j++){
                Imgproc.rectangle(mRgbaROI, noseArray[i].tl(),noseArray[i].br(), new Scalar(150, 150, 150, 255), 3);
                //podpisanie prostokąta
                Imgproc.putText(mRgbaROI,"nos",new Point(noseArray[i].x-5,noseArray[i].y-5),Core.FONT_ITALIC,1,new Scalar(0,100,0,255),2);
                noseX = noseArray[i].tl().x-noseArray[i].br().x;
                noseY = noseArray[i].tl().y-noseArray[i].br().y;
            }

        }
        return mRgba;
    }



    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.options, menu);
        return true;
    }
    // Obsługa opcji dostępnych w rozwijanym menu - popup menu
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.frontCamera:
                // zmiana kamery na przednia
                if(javaCameraView!=null){
                    javaCameraView.disableView();
                }
                cameraID=1;
                color=0;
                javaCameraView.setVisibility(SurfaceView.VISIBLE);
                javaCameraView.setCvCameraViewListener(this);
                javaCameraView.setCameraIndex(cameraID);
                mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
                System.out.println("Start przycisk");

                return true;
            case R.id.backCamera:
                //zmiana kamery na tylna
                System.out.println("Stop przycisk");
                if(javaCameraView!=null){
                    javaCameraView.disableView();
                }
                cameraID=0;
                color=0;
                javaCameraView.setVisibility(SurfaceView.VISIBLE);
                javaCameraView.setCvCameraViewListener(this);
                javaCameraView.setCameraIndex(cameraID);
                mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);

                return true;
            case R.id.grayScale:
                System.out.println("Skala szarości");
                color=1;

            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    public void onClick(View view) {
        switch(view.getId()) {
            case R.id.capture:
                System.out.println("Przycisk CAPTURE działa");
                img1 = mRgbaROI;

                // DESKRYPTOR
                /*descriptors1 = new Mat();
                keypoints1 = new MatOfKeyPoint();

                FeatureDetector detector1 = FeatureDetector.create(FeatureDetector.ORB);
                detector1.detect(img1, keypoints1);
                DescriptorExtractor extractor1 = DescriptorExtractor.create(DescriptorExtractor.ORB);
                extractor1.compute(img1, keypoints1,descriptors1);*/
                patternList[0]=Math.abs(eye1X);
                patternList[1]= Math.abs(eye1Y);
                patternList[2]= Math.abs(eye2X);
                patternList[3]= Math.abs(eye2Y);
                patternList[4]= Math.abs(mouthX);
                patternList[5]=  Math.abs(mouthY);
                patternList[6]= Math.abs(noseX);
                patternList[7]=Math.abs(noseY) ;

                d1 = Math.sqrt(Math.pow(patternList[0] - patternList[2], 2) + Math.pow(patternList[1] - patternList[3], 2));


                break;

            case R.id.check:
                img2 = mRgbaROI;
                //DESKRYPTOR
              /*  descriptors2 = new Mat();
                keypoints2 = new MatOfKeyPoint();
                FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
                detector.detect(img2, keypoints2);
                DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
                extractor.compute(img2, keypoints2,descriptors2);*/

                checkList[0]=Math.abs(eye1X);
                checkList[1]= Math.abs(eye1Y);
                checkList[2]= Math.abs(eye2X);
                checkList[3]= Math.abs(eye2Y);
                checkList[4]= Math.abs(mouthX);
                checkList[5]= Math.abs(mouthY);
                checkList[6]= Math.abs(noseX);
                checkList[7]=Math.abs(noseY);

                d2 = Math.sqrt(Math.pow(checkList[0] - checkList[2], 2) + Math.pow(checkList[1] - checkList[3], 2));


                int counter=0;
                for(int i=0;i<8;i++){
                    if(Math.abs(patternList[i]-checkList[i])<11){
                        counter++;
                        System.out.println("Stan licznika: "+counter);
                        System.out.println("Odległość między oczami: "+d1+" drugi "+d2);
                    }
                }

                if(counter>=5){
                   // Imgproc.putText(mRgba,"Tomasz",new Point(faceX-5,faceY-5),Core.FONT_ITALIC,1,new Scalar(0,255,0,255),4);
                    person="Tomasz";
                }

                // DOPASOWYWANIE
               /* matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
                //matcher should include 2 different image's descriptors
                MatOfDMatch  matches = new MatOfDMatch();
                matcher.match(descriptors1,descriptors2,matches);
                //feature and connection colors
                Scalar RED = new Scalar(255,0,0);
                Scalar GREEN = new Scalar(0,255,0);
                //output image
                Mat outputImg = new Mat();
                MatOfByte drawnMatches = new MatOfByte();
                //this will draw all matches, works fine
                Features2d.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                        outputImg, GREEN, RED,  drawnMatches, Features2d.NOT_DRAW_SINGLE_POINTS);
                System.out.println(matches.size());*/

               /* //zapisywanie i wyświetlanie działania deskryptora
                Bitmap imageMatched = Bitmap.createBitmap(outputImg.cols(), outputImg.rows(), Bitmap.Config.RGB_565);//need to save bitmap
                Utils.matToBitmap(outputImg, imageMatched);
                // zapis do pliku
                saveToInternalStorage(imageMatched);
                setContentView(R.layout.obraz);
                image = (ImageView) findViewById(R.id.imageView);
                image.setImageBitmap(imageMatched);*/


                System.out.println("Przycisk CHECK działa");

                break;

        }
    }

    private String saveToInternalStorage(Bitmap bitmapImage){
        ContextWrapper cw = new ContextWrapper(getApplicationContext());
        // path to /data/data/yourapp/app_data/imageDir
        File directory = cw.getDir("imageDir", Context.MODE_PRIVATE);
        // Create imageDir
        File mypath=new File(directory,"profile.jpg");

        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream(mypath);
            // Use the compress method on the BitMap object to write image to the OutputStream
            bitmapImage.compress(Bitmap.CompressFormat.PNG, 100, fos);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                fos.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        System.out.println("Zapisałem obraz");
        System.out.println(directory.getAbsolutePath());
        return directory.getAbsolutePath();
    }
}
