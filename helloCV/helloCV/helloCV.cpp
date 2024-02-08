// helloCV.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <opencv2/opencv.hpp>
//using namespace cv;

#define _USE_MATH_DEFINES // for C++
#include <math.h>
using namespace std;

cv::Mat myrotate(cv::Mat img, int deg);
int main()
{
    //Point pt1,pt2;
    //pt1.x = 10;
    //pt1.y = 20;

    //pt2.x = 10;
    //pt2.y = 10;
    //Point pt3 = pt1 + pt2;
    //cout << pt3 << endl;
    //
    /////////////////////////////////
    //// size  
    ////////////////////////////////
    //Size sz1, sz2(10, 20); //size 는 shape 
    //sz1.width = 5;
    //sz1.height = 10; 
    //cout << sz1 + sz2 << endl;
    //cout << sz1.area() << endl;

    //Rect rc1(10,10,60,40);
    //Rect rc2;
    //Rect rc3 = rc2 + sz1;

    //Mat mat1(2, 3, CV_8UC1); // 1채널 매트릭스 생성 
    
    ///////////////////////////////
    // size  
    //////////////////////////////

    std::cout << "Hello World!\n<<CV_VERSION<<std::endl";
    auto img = cv::imread("images/lena.jpg");
    cout << img.at<uchar>(50,50) << endl;
   

    //cv::Mat rotate_img;

    cv::Mat rotate_img = myrotate(img, 40);

    //rotate(img,rotate_img, cv::ROTATE_90_COUNTERCLOCKWISE);
    imshow("img", img);
    imshow("rotate_img", rotate_img);
    cv::waitKey();

}

cv::Mat myrotate(cv::Mat img, int deg) {
    int height = img.rows;
    int width = img.cols;
    cv::Size s = img.size();

    int newX = 0, newY = 0;
    cv::Mat result = cv::Mat::zeros(s, CV_8UC3);
    int center_x = int(s.width / 2);
    int center_y = int(s.height / 2);


    for (int y = 0; y < s.height; y++) {
        for (int x = 0; x < s.width; x++) {
            newX = int(cos(deg / 180 * M_PI) * (x) + sin(deg / 180 * M_PI) * (y));
            newY = int(-sin(deg / 180 * M_PI) * (x) + cos(deg / 180 * M_PI) * (y));
            //경계선 검사 
            if ((newX < 0) || (newX >= s.width) || (newY < 0) || (newY >= s.height)) {
                continue;
            }
            result.at<uchar>(newY, newX) = img.at<uchar>(y, x);
            cout << img.at<uchar>(y, x) << endl;

        }

    }
    return result;
}


        //height, width = img.shape
        //_img = np.zeros((height, width), dtype = np.uint8)
        //for y in range(height) :
        //    for x in range(width) :
        //        X = int(np.cos(np.radians(degree)) * x + np.sin(np.radians(degree)) * y)
        //        Y = int(-np.sin(np.radians(degree)) * x + np.cos(np.radians(degree)) * y)
        //        #경계선 검사
        //        if (X < 0) | (X >= width) | (Y < 0) | (Y >= height) :
        //            continue
        //            _img[Y, X] = img[y, x]

        //            return _img


// 프로그램 실행: <Ctrl+F5> 또는 [디버그] > [디버깅하지 않고 시작] 메뉴
// 프로그램 디버그: <F5> 키 또는 [디버그] > [디버깅 시작] 메뉴

// 시작을 위한 팁: 
//   1. [솔루션 탐색기] 창을 사용하여 파일을 추가/관리합니다.
//   2. [팀 탐색기] 창을 사용하여 소스 제어에 연결합니다.
//   3. [출력] 창을 사용하여 빌드 출력 및 기타 메시지를 확인합니다.
//   4. [오류 목록] 창을 사용하여 오류를 봅니다.
//   5. [프로젝트] > [새 항목 추가]로 이동하여 새 코드 파일을 만들거나, [프로젝트] > [기존 항목 추가]로 이동하여 기존 코드 파일을 프로젝트에 추가합니다.
//   6. 나중에 이 프로젝트를 다시 열려면 [파일] > [열기] > [프로젝트]로 이동하고 .sln 파일을 선택합니다.
