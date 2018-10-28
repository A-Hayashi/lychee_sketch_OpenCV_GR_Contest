/* Face detection example with OpenCV */
/* Public Domain             */

#include <Arduino.h>
#include <Camera.h>
#include <opencv.hpp>
#include <DisplayApp.h>

// To monitor realtime on PC, you need DisplayApp on following site.
// Connect USB0(not for mbed interface) to your PC
// https://os.mbed.com/users/dkato/code/DisplayApp/
#include "mbed.h"
#include "SdUsbConnect.h"

#include "constants.h"
#include "findEyeCenter.h"
#include "findEyeCorner.h"

#define IMAGE_HW 320
#define IMAGE_VW 240
using namespace cv;

/* FACE DETECTOR Parameters */
#define DETECTOR_SCALE_FACTOR (1.05)
#define DETECTOR_MIN_NEIGHBOR (2)
#define DETECTOR_MIN_SIZE     (80)
#define FACE_DETECTOR_MODEL     "/storage/lbpcascade_frontalface.xml"

static Camera camera(IMAGE_HW, IMAGE_VW);
static DisplayApp  display_app;
static CascadeClassifier detector_classifier;

#define H_MAX 30
#define H_MIN 0
#define S_MAX 255
#define S_MIN 50
#define V_MAX 255
#define V_MIN 50

uint8_t bgr_buf [3 * IMAGE_HW * IMAGE_VW] __attribute((section("NC_BSS"),aligned(32)));
uint8_t hsv_buf [3 * IMAGE_HW * IMAGE_VW] __attribute((section("NC_BSS"),aligned(32)));
uint8_t gray_buf [1 * IMAGE_HW * IMAGE_VW] __attribute((section("NC_BSS"),aligned(32)));
uint8_t mask_buf [1 * IMAGE_HW * IMAGE_VW] __attribute((section("NC_BSS"),aligned(32)));

Scalar red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);
Scalar yellow = red + green;
Scalar sky = green + blue;
Scalar white = Scalar::all(255);
Scalar pink = Scalar(154, 51, 255);

Timer t;

void setup() {
    pinMode(PIN_LED_GREEN, OUTPUT);
    pinMode(PIN_LED_RED, OUTPUT);
    pinMode(PIN_SW0, INPUT);
    pinMode(PIN_SW1, INPUT);

    // Camera
    camera.begin();

    // SD & USB
    SdUsbConnect storage("storage");
    storage.wait_connect();

    // Load the cascade classifier file
    detector_classifier.load(FACE_DETECTOR_MODEL);

    if (detector_classifier.empty()) {
        digitalWrite(PIN_LED_RED, HIGH); // Error
        CV_Assert(0);
        mbed_die();
    }

    t.reset();
    t.start();
}


void FaceDetect(Mat &img_gray, Rect &face_roi)
{
   if (detector_classifier.empty()) {
		digitalWrite(PIN_LED_RED, HIGH); // Error
	}

	// Perform detected the biggest face
	std::vector<Rect> rect_faces;
	detector_classifier.detectMultiScale(img_gray, rect_faces,
										 DETECTOR_SCALE_FACTOR,
										 DETECTOR_MIN_NEIGHBOR,
										 CASCADE_SCALE_IMAGE | CASCADE_FIND_BIGGEST_OBJECT,
										 Size(DETECTOR_MIN_SIZE, DETECTOR_MIN_SIZE));

	if (rect_faces.size() > 0) {
		// A face is detected
		face_roi = rect_faces[0];
	} else {
		// No face is detected, set an invalid rectangle
		face_roi.x = -1;
		face_roi.y = -1;
		face_roi.width = -1;
		face_roi.height = -1;
	}

	if (face_roi.width > 0 && face_roi.height > 0) {   // A face is detected
		digitalWrite(PIN_LED_GREEN, HIGH);
		printf("Detected a face X:%d Y:%d W:%d H:%d\n",face_roi.x, face_roi.y, face_roi.width, face_roi.height);
		digitalWrite(PIN_LED_GREEN, LOW);
	} else {
	}
}

void SkinDetect(Mat &img_hsv, Mat &img_gray, vector<Point> &contour, Point2f &center, vector<Point> &contour2)
{
//	medianBlur(img_hsv, img_hsv, 3);

	Scalar s_min = Scalar(H_MIN, S_MIN, V_MIN);
	Scalar s_max = Scalar(H_MAX, S_MAX, V_MAX);
	inRange(img_hsv, s_min, s_max, img_gray);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(img_gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	if(contours.size() > 0){
		vector<Moments> mu(contours.size());
		for (size_t i = 0; i < contours.size(); i++){
			mu[i] = moments(contours[i], false);
		}
		vector<Point2f> mc(contours.size()); //get centers
		for (size_t i = 0; i < contours.size(); i++)
		{
			mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
		}
		size_t indexOfBiggestContour = -1;
		size_t sizeOfBiggestContour = 0;
		size_t indexOfSecondContour = -1;
		size_t sizeOfSecondContour = 0;

		for (size_t i = 0; i < contours.size(); i++) {
			size_t area = contourArea(contours[i]);
			if (area > sizeOfBiggestContour) {
				sizeOfBiggestContour = area;
				indexOfBiggestContour = i;
			}else if(area > sizeOfSecondContour){
				sizeOfSecondContour = area;
				indexOfSecondContour = i;
			}
		}
		contour = contours[indexOfBiggestContour];
		center = mc[indexOfBiggestContour];

		if(indexOfSecondContour>0){
			contour2 = contours[indexOfSecondContour];
		}else{
			contour2.clear();
		}
	}else{
		contour.clear();
		contour2.clear();
		center.x = -1;
		center.y = -1;
	}

}

static uint8_t flag = 0;
void loop(){
    Mat img_raw(IMAGE_VW, IMAGE_HW, CV_8UC2, camera.getImageAdr());
    Mat img_gray(IMAGE_VW, IMAGE_HW, CV_8UC1, gray_buf);
    Mat img_mask(IMAGE_VW, IMAGE_HW, CV_8UC1, mask_buf);
    Mat img_bgr(IMAGE_VW, IMAGE_HW, CV_8UC3, bgr_buf);
    Mat img_hsv(IMAGE_VW, IMAGE_HW, CV_8UC3, hsv_buf);

    vector<Point> contour, contour2;
    Point2f center;
    cvtColor(img_raw, img_bgr, COLOR_YUV2BGR_YUYV); //covert from YUV to BGR
    cvtColor(img_bgr, img_hsv, COLOR_BGR2HSV); //covert from YUV to BGR
    SkinDetect(img_hsv, img_mask, contour, center, contour2);


	cvtColor(img_raw, img_gray, COLOR_YUV2GRAY_YUYV); //covert from YUV to GRAY
	Mat img_gray_roi;
	Rect rect, rect2, face_roi;

	if(contour.size() > 0){
		Rect rect_base(0, 0, IMAGE_HW, IMAGE_VW);
		rect = boundingRect(contour);
		rect2 = rect;
		Size deltaSize(rect.width*0.5f, rect.height*0.3f);
		Point offset(deltaSize.width/2, deltaSize.height/2 + rect.height*0.2f);
		rect += deltaSize;
		rect -= offset;
		rect &= rect_base;
		img_gray_roi = img_gray(rect);

		// Detect a face in the frame
	    FaceDetect(img_gray_roi, face_roi);
	    face_roi += Point(rect.x, rect.y);
	}

    if(digitalRead(PIN_SW0)==0){
    	flag = 0x01;
    }else if(digitalRead(PIN_SW1)==0){
    	flag = 0x02;
    }else{
    	flag = 0x00;
    }

    Mat img_bgr2(IMAGE_VW, IMAGE_HW, CV_8UC3, hsv_buf);
    img_bgr2 = Mat::zeros(IMAGE_VW, IMAGE_HW, CV_8UC3);

    if(flag==0x00){
  		static int nose = 0;
    	static int c = 0;
    	if (face_roi.width > 0 && face_roi.height > 0){
    		rectangle(img_bgr, face_roi, red, 1);
    		circle(img_bgr, Point(face_roi.x+face_roi.width/2, face_roi.y+face_roi.height/2), 5, red, -1, 8, 0);
    		if(t.read_ms()>500){
    			t.reset();
    			nose = face_roi.x+face_roi.width/2;
    			c = center.x;
    		}
    	}
		if(nose < c){
			putText(img_bgr, "Right", Point(10,50), FONT_HERSHEY_SIMPLEX, 1.2, red, 2, LINE_AA);
		}else if(nose > c){
			putText(img_bgr, "Left", Point(10,50), FONT_HERSHEY_SIMPLEX, 1.2, red, 2, LINE_AA);
		}
    	if(contour.size() > 0){
			rectangle(img_bgr, rect, blue, 1);
			rectangle(img_bgr, rect2, pink, 1);
			polylines(img_bgr, contour, true, green, 1, 8);
			circle(img_bgr, center, 5, green, -1, 8, 0);
    	}

//    	if(contour2.size() > 0){
//			polylines(img_bgr, contour2, true, green, 1, 8);
//    	}
    	size_t jpegSize = camera.createJpeg(IMAGE_HW, IMAGE_VW, img_bgr.data, Camera::FORMAT_RGB888);
		display_app.SendJpeg(camera.getJpegAdr(), jpegSize);
	}else if(flag==0x01){
		img_bgr.copyTo(img_bgr2, img_mask);
		if (face_roi.width > 0 && face_roi.height > 0){
			rectangle(img_bgr, face_roi, red, 2);
		}
		size_t jpegSize = camera.createJpeg(IMAGE_HW, IMAGE_VW, img_bgr2.data, Camera::FORMAT_RGB888);
		display_app.SendJpeg(camera.getJpegAdr(), jpegSize);
	}else if(flag==0x02){
		if (face_roi.width > 0 && face_roi.height > 0){
			rectangle(img_bgr, face_roi, red, 2);
		}
		size_t jpegSize = camera.createJpeg(IMAGE_HW, IMAGE_VW, img_mask.data, Camera::FORMAT_GRAY);
		display_app.SendJpeg(camera.getJpegAdr(), jpegSize);
	}


}


