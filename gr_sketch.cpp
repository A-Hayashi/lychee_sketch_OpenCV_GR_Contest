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
#define DETECTOR_SCALE_FACTOR (1.99)
#define DETECTOR_MIN_NEIGHBOR (3)
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

Scalar red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);
Scalar yellow = red + green;
Scalar sky = green + blue;
Scalar white = Scalar::all(255);
Scalar pink = Scalar(154, 51, 255);

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

void SkinDetect(Mat &img_hsv, Mat &img_gray, vector<Point> &contour, Point2f &center)
{
	medianBlur(img_hsv, img_hsv, 3);

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
		for (size_t i = 0; i < contours.size(); i++) {
			if (contours[i].size() > sizeOfBiggestContour) {
				sizeOfBiggestContour = contours[i].size();
				indexOfBiggestContour = i;
			}
		}
		contour = contours[indexOfBiggestContour];
		center = mc[indexOfBiggestContour];
	}else{
		contour.clear();
		center.x = -1;
		center.y = -1;
	}

}

static uint8_t flag = 0;
void loop(){
    Mat img_raw(IMAGE_VW, IMAGE_HW, CV_8UC2, camera.getImageAdr());
    Mat img_gray(IMAGE_VW, IMAGE_HW, CV_8UC1, gray_buf);
    Mat img_bgr(IMAGE_VW, IMAGE_HW, CV_8UC3, bgr_buf);
    Mat img_hsv(IMAGE_VW, IMAGE_HW, CV_8UC3, hsv_buf);

    cvtColor(img_raw, img_gray, COLOR_YUV2GRAY_YUYV); //covert from YUV to GRAY

    // Detect a face in the frame
    Rect face_roi;
    FaceDetect(img_gray, face_roi);

    cvtColor(img_raw, img_bgr, COLOR_YUV2BGR_YUYV); //covert from YUV to BGR
    cvtColor(img_bgr, img_hsv, COLOR_BGR2HSV); //covert from YUV to BGR

    vector<Point> contour;
    Point2f center;
    SkinDetect(img_hsv, img_gray, contour, center);

//	rect.x -= 40;
//	rect.y -= 40;
//	rect.width += 80;
//	rect.height += 80;

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
    	rectangle(img_bgr, Point(face_roi.x, face_roi.y), Point(face_roi.x + face_roi.width, face_roi.y + face_roi.height), red, 2);
    	if(contour.size() > 0){
    		Rect rect = boundingRect(contour);
			rectangle(img_bgr, rect, blue, 2);
			polylines(img_bgr, contour, true, sky, 2, 8);
			circle(img_bgr, center, 5, red, -1, 8, 0);
    	}
    	size_t jpegSize = camera.createJpeg(IMAGE_HW, IMAGE_VW, img_bgr.data, Camera::FORMAT_RGB888);
		display_app.SendJpeg(camera.getJpegAdr(), jpegSize);
    }else if(flag==0x01){
    	img_bgr.copyTo(img_bgr2, img_gray);
    	rectangle(img_bgr2, Point(face_roi.x, face_roi.y), Point(face_roi.x + face_roi.width, face_roi.y + face_roi.height), red, 2);
    	size_t jpegSize = camera.createJpeg(IMAGE_HW, IMAGE_VW, img_bgr2.data, Camera::FORMAT_RGB888);
		display_app.SendJpeg(camera.getJpegAdr(), jpegSize);
    }else if(flag==0x02){
    	rectangle(img_gray, Point(face_roi.x, face_roi.y), Point(face_roi.x + face_roi.width, face_roi.y + face_roi.height), red, 2);
    	size_t jpegSize = camera.createJpeg(IMAGE_HW, IMAGE_VW, img_gray.data, Camera::FORMAT_GRAY);
		display_app.SendJpeg(camera.getJpegAdr(), jpegSize);
    }
}


