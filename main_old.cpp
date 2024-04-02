// Adaptation of openfeed.py 
//     Author: D. Schmidt 3-15-2024
//
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int main()
{
    
    std::string url = "rtsp://service:Bosch123!@172.31.0.8:554?line=1&inst=1";
    unsigned char m_enable = 1;

    cv::VideoCapture capture(url);

    if (!capture.isOpened()) {
        std::cout << "Initial capture error, aborting." << std::endl;
	return 0;
    }
    
    

    cv::namedWindow("TEST", WINDOW_AUTOSIZE);
    cv::Mat frame;

    while(m_enable) {
        if (!capture.read(frame)) {
            std::cout << "Error reading frame" << std::endl;
        }
        cv::imshow("TEST", frame);

        cv::waitKey(25);
    }  
    return 0;

}
