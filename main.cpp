// Adaptation of openfeed.py 
// Author: 
//    D. Schmidt 3-15-2024
//
// Build with:
//    g++ main.cpp -o output `pkg-config --cflags --libs opencv4` -std=c++20
//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <iterator>
#include <fstream>
#include <vector>
#include <algorithm> // for std::copy
#include <iostream>
#include <sstream>
#include <string>


using namespace cv;

class Stream {
    public:
        unsigned char frame_width = 0;
        unsigned char frame_height = -1;
        unsigned char fps = -1;
        unsigned char threshold = 0.52;
        unsigned char refresh = -1;
        unsigned char boost = 1.25;       
        };
        
// Set up the detector with default parameters.
//SimpleBlobDetector detector;

int main() {
     
     std::vector<std::string> classes;
     std::string file = "/home/charlie/repos/buoy_vision/ssd_mobilenet/coco_names.txt";
     std::ifstream ifs(file.c_str());
     if (!ifs.is_open()){
         CV_Error(Error::StsError, "File " + file + " not found");
     }

     std::string line;
     while (std::getline(ifs, line))
        {
        classes.push_back(line);
        std::cout<<line<<std::endl;
        }
     


    //Load model weights
    std::string weights = "/home/charlie/repos/buoy_vision/ssd_mobilenet/frozen_inference_graph.pb";
    std::string pbtxt = "/home/charlie/repos/buoy_vision/ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
    double scale = 1.0/127.5;
    int width = 320;
    int height = 320;
    Scalar avg = 127.5;

    dnn::Net net = cv::dnn::readNetFromTensorflow(weights, pbtxt);
  

    // Main video loop
    std::cout << "Beginning image classification tasks." << std::endl;
    std::string url = "rtsp://service:Bosch123!@172.31.0.8:554?line=1&inst=1";
    unsigned char m_enable = 1;
    cv::VideoCapture capture(url);

    if (!capture.isOpened()) {
        std::cout << "Initial capture error, aborting." << std::endl;
	return 0;
    }
    std::cout << "RTSP stream acquired, generating window." << std::endl;
    
    cv::namedWindow("TEST", cv::WINDOW_AUTOSIZE);
    cv::Mat frame,blob;
    cv::Mat output;
    Point classIdPoint;
    double confidence;
    int classId = 0;
    std::string label = "";

    while(m_enable) {
        if (!capture.read(frame)) {
            std::cout << "Error reading frame" << std::endl;
        }

	dnn::blobFromImage(frame, blob, scale, cv::Size(width,height), avg, false, false);
	net.setInput(blob);
	output = net.forward();
        minMaxLoc(output.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
        classId = classIdPoint.x;
	std::cout << classId << " " << std::endl;
	//label = format("%s: %.4f", (classes.empty() ? format("Class #%d", classId).c_str() : classes[classId].c_str()),confidence);
        //putText(frame, label, Point(0, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	
        cv::imshow("TEST", frame);

        cv::waitKey(25);
    }  
    return 0;

}
