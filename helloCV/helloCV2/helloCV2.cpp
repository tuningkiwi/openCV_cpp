// helloCV.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <opencv2/opencv.hpp>

void rotate(cv::Mat img, cv::Mat* rotate_img, int deg);

int main()
{
    std::cout << "Hello World!\n<<CV_VERSION<<std::endl";
    cv::Mat img = cv::imread("images/lena.jpg");
    cv::Mat rotate_img; 
    
    rotate(img, &rotate_img, 40);
    imshow("image", img);
    imshow("rotate", rotate_img);
    return 0; 

}


void rotate(cv::Mat img, cv::Mat* rotate_img, int deg) {
    cv::Size s = img.size();

    //double theta = (CV_PI* deg)/180.0 
    double angleRadians = (CV_PI * deg) / 180.0;
    double cosTheta = cos(angleRadians); 
    double sinTheta = sin(angleRadians);

    cv::Size s = img.size();

    rotate_img = cv::Mat((s, img.type(), cv::Scalar(0, 0, 0)));
    cv::Point2f center(static_cast<float>(img.cols / 2), static_cast<float>(img.rows / 2));

    for (size_t y = 0; y < rotate_img.rows; y++) {
        for (size_t x = 0; x < rotate_img.cols; x++) {
            int originX = static_cast<int>(std::round((x - center.x) * cosTheta - (y - center.y)*sinTheta));
            int originY = static_cast<int>(std::round((x - center.x) * sinTheta + (y - center.y)*cosTheta));
            if (originX >= 0 && originX < img.cols && originY >= 0 && originY < img.rows) {
                rotate_img.at<cv::Vec3b>(y, x) = img.at<cv::Vec3b>(originY, originX);
            }
             
        }
    }

}