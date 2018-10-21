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

#define IMAGE_HW 320
#define IMAGE_VW 240
using namespace cv;

/* FACE DETECTOR Parameters */
#define DETECTOR_SCALE_FACTOR (2)
#define DETECTOR_MIN_NEIGHBOR (2)
#define DETECTOR_MIN_SIZE     (30)
#define FACE_DETECTOR_MODEL     "/storage/lbpcascade_frontalface.xml"


static Camera camera(IMAGE_HW, IMAGE_VW);
static DisplayApp  display_app;
static CascadeClassifier detector_classifier;

void setup() {
    pinMode(PIN_LED_GREEN, OUTPUT);
    pinMode(PIN_LED_RED, OUTPUT);

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

void loop(){
    Mat img_raw(IMAGE_VW, IMAGE_HW, CV_8UC2, camera.getImageAdr());

    Mat src;
    cvtColor(img_raw, src, COLOR_YUV2GRAY_YUYV); //covert from YUV to GRAY

    // Detect a face in the frame
    Rect face_roi;
    if (detector_classifier.empty()) {
        digitalWrite(PIN_LED_RED, HIGH); // Error
    }

    // Perform detected the biggest face
    std::vector<Rect> rect_faces;
    detector_classifier.detectMultiScale(src, rect_faces,
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

    Mat dst;
    cvtColor(img_raw, dst, COLOR_YUV2BGR_YUYV); //covert from YUV to BGR
    Scalar red(0, 0, 255), green(0, 255, 0), blue(255, 0, 0);

    rectangle(dst, Point(face_roi.x, face_roi.y), Point(face_roi.x + face_roi.width, face_roi.y + face_roi.height), red, 2);
    size_t jpegSize = camera.createJpeg(IMAGE_HW, IMAGE_VW, dst.data, Camera::FORMAT_RGB888);
    display_app.SendJpeg(camera.getJpegAdr(), jpegSize);
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
  cv::Mat faceROI = frame_gray(face);
  cv::Mat debugFace = faceROI;

  if (kSmoothFaceImage) {
    double sigma = kSmoothFaceFactor * face.width;
    GaussianBlur( faceROI, faceROI, cv::Size( 0, 0 ), sigma);
  }
  //-- Find eye regions and draw them
  int eye_region_width = face.width * (kEyePercentWidth/100.0);
  int eye_region_height = face.width * (kEyePercentHeight/100.0);
  int eye_region_top = face.height * (kEyePercentTop/100.0);
  cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
  cv::Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                          eye_region_top,eye_region_width,eye_region_height);

  //-- Find Eye Centers
  cv::Point leftPupil;
  cv::Point rightPupil;

//  leftPupil = findEyeCenter(faceROI,leftEyeRegion,"Left Eye");
//  rightPupil = findEyeCenter(faceROI,rightEyeRegion,"Right Eye");

  // get corner regions
  cv::Rect leftRightCornerRegion(leftEyeRegion);
  leftRightCornerRegion.width -= leftPupil.x;
  leftRightCornerRegion.x += leftPupil.x;
  leftRightCornerRegion.height /= 2;
  leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
  cv::Rect leftLeftCornerRegion(leftEyeRegion);
  leftLeftCornerRegion.width = leftPupil.x;
  leftLeftCornerRegion.height /= 2;
  leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
  cv::Rect rightLeftCornerRegion(rightEyeRegion);
  rightLeftCornerRegion.width = rightPupil.x;
  rightLeftCornerRegion.height /= 2;
  rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
  cv::Rect rightRightCornerRegion(rightEyeRegion);
  rightRightCornerRegion.width -= rightPupil.x;
  rightRightCornerRegion.x += rightPupil.x;
  rightRightCornerRegion.height /= 2;
  rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
  rectangle(debugFace,leftRightCornerRegion,200);
  rectangle(debugFace,leftLeftCornerRegion,200);
  rectangle(debugFace,rightLeftCornerRegion,200);
  rectangle(debugFace,rightRightCornerRegion,200);
  // change eye centers to face coordinates
  rightPupil.x += rightEyeRegion.x;
  rightPupil.y += rightEyeRegion.y;
  leftPupil.x += leftEyeRegion.x;
  leftPupil.y += leftEyeRegion.y;
  // draw eye centers
//  circle(debugFace, rightPupil, 3, 1234);
//  circle(debugFace, leftPupil, 3, 1234);

  //-- Find Eye Corners
  if (kEnableEyeCorner) {
    cv::Point2f leftRightCorner;
    cv::Point2f leftLeftCorner;
    cv::Point2f rightLeftCorner;
    cv::Point2f rightRightCorner;

//    leftRightCorner = findEyeCorner(faceROI(leftRightCornerRegion), true, false);
    leftRightCorner.x += leftRightCornerRegion.x;
    leftRightCorner.y += leftRightCornerRegion.y;
//    leftLeftCorner = findEyeCorner(faceROI(leftLeftCornerRegion), true, true);
    leftLeftCorner.x += leftLeftCornerRegion.x;
    leftLeftCorner.y += leftLeftCornerRegion.y;
//    rightLeftCorner = findEyeCorner(faceROI(rightLeftCornerRegion), false, true);
    rightLeftCorner.x += rightLeftCornerRegion.x;
    rightLeftCorner.y += rightLeftCornerRegion.y;
//    rightRightCorner = findEyeCorner(faceROI(rightRightCornerRegion), false, false);
    rightRightCorner.x += rightRightCornerRegion.x;
    rightRightCorner.y += rightRightCornerRegion.y;
//    circle(faceROI, leftRightCorner, 3, 200);
//    circle(faceROI, leftLeftCorner, 3, 200);
//    circle(faceROI, rightLeftCorner, 3, 200);
//    circle(faceROI, rightRightCorner, 3, 200);
  }

//  imshow(face_window_name, faceROI);
}

void detectAndDisplay( cv::Mat frame_gray ) {
  std::vector<cv::Rect> faces;

  if (faces.size() > 0) {
    findEyes(frame_gray, faces[0]);
  }
}
