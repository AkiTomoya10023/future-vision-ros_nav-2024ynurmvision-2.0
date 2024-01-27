//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include "inference.h"
//
//using namespace yolox_detector;
//
//void image_detect_test(cv::Mat& src, ArmorDetector& detector){
//    cv::Mat res = detector.detect(src);
//    cv::imwrite("img_test.jpg", res);
//}
//
//void vidoe_detect_test(const string input_path, ArmorDetector& detector){
//    cv::VideoCapture video_cap;
//    cv::Mat src;
//    video_cap.open(input_path);
//    assert(video_cap.isOpened());
//    while(video_cap.read(src)) {
//        cv::Mat res = detector.detect(src);
//        std::cout << "detect running" <<std::endl;
//        cv::namedWindow("network_output", 0);
//        cv::imshow("network_output", res);
//        cv::waitKey(1);
//    }
//    video_cap.release();
//
//}

//int main(int argc, char** argv) {
//    ArmorDetector detector;
//    if (argc == 5 && std::string(argv[2]) == "-i") {
//        const std::string engine_file_path {argv[1]};
//        if (!(detector.initModel(engine_file_path))){
//            std::cerr << " Init model FAILED." << std::endl;
//            return -1;
//        }
//        try {
//            const std::string input_path{argv[3]};
//            if (std::string(argv[4]) == "-image") {
//                cv::Mat src = cv::imread(input_path);
//                image_detect_test(src, detector);
//            } else if (std::string(argv[4]) == "-video") {
//                vidoe_detect_test(input_path, detector);
//            }
//        }
//        catch (std::exception e){
//            std::cout<<e.what()<<std::endl;
//        }
//    } else {
//        std::cerr << "arguments not right!" << std::endl;
//        return -1;
//    }
//    return 0;
//}
