// helloCV.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#define _USE_MATH_DEFINES // for C++
#include <math.h>
#include <stdio.h>
#define FILENAME  "test.txt"

using namespace std;
using namespace cv;

int color_equalization(void);
int color_inrange(void);
void on_hue_changed(int, void*);
int printCodeList(void);
int color_inrange_v2(void);
void nothing(int, void*);
int color_backprojection(void);
int filter_embossing(void);
int histogram_equalization(void);
Mat getGrayHistImage(const Mat& hist);
Mat calcGrayHist(const Mat& img);
int blurring_mean(void); 
int blurring_gaussian(void); 
int sharpen(void); 

int main()
{
    int codeNum = 0, result = 0;

    while (true) {
        //코드 리스트 출력하기 
        if ((result = printCodeList()) != 0) {
            cout << "Print Code List Error" << endl;
            return -1; 
        }
        cout << "\n코드 ID를 입력하세요 : ";
        cin >> codeNum;

        switch (codeNum) {
            case 104: color_equalization(); break;
            case 105: color_inrange(); break;
            case 1052: color_inrange_v2(); break;
            case 106: color_backprojection(); break;
            case 71: filter_embossing(); break;
            case 72: blurring_mean(); break;
            case 73: blurring_gaussian(); break;
            case 74: sharpen(); break; 
            case 510: histogram_equalization(); break;
            
            case 0: cout << "Program is closed ..." << endl; return 0;
            default: cout << "Wrong Code ID, Retry!!!!!!!!" << endl; break;
        }
    
    }
    
}

int sharpen(void) {
    Mat src = imread("images/rose.bmp", IMREAD_GRAYSCALE);

    if (src.empty()) {
        cerr << "image load failed" << endl;
        return -1;
    }

    imshow("src", src);

    for (int sigma = 1; sigma <= 5; sigma++) {
        Mat blurred; 
        GaussianBlur(src, blurred, Size(), (double)sigma);

        float alpha = 1.f; 
        Mat dst = (1 + alpha) * src - alpha * blurred;

        String dstTitle = format("sigma=%d", sigma);
        imshow(dstTitle, dst);
    }

    waitKey();
    destroyAllWindows();
}

int blurring_gaussian(void) {
    Mat src = imread("images/rose.bmp", IMREAD_GRAYSCALE); 

    if (src.empty()) {
        cerr << "image load failed" << endl;
        return -1;
    }

    imshow("src", src);

    Mat dst; 
    for (int sigma = 1; sigma <= 5; sigma++) {
        GaussianBlur(src, dst, Size(), (double)sigma); 

        cout << "Gaussian Kernel\n" << getGaussianKernel(9, sigma) << endl; 
        String dstTitle = format("sigma=%d", sigma);
        imshow(dstTitle, dst);
    }

    waitKey();
    destroyAllWindows(); 

}

int blurring_mean(void) {
    Mat src = imread("images/rose.bmp", IMREAD_GRAYSCALE);

    if (src.empty()) {
        cerr << "image load failed" << endl;
        return -1; 
    }

    imshow("src", src); 

    Mat dst; 
    for (int ksize = 3; ksize <= 7; ksize += 2) {
        blur(src, dst, Size(ksize,ksize));// kernel = src /(ksize.width*ksiz.height)

        String desc = format("Mean:%dx%d", ksize, ksize); 
        imshow(desc, dst);
    }
    waitKey();
    destroyAllWindows(); 
}

int histogram_equalization(void) {
    Mat src = imread("images/hawkes.bmp", IMREAD_GRAYSCALE); 

    if (src.empty()) {
        cerr << "image laod failed" << endl;
        return -1; 
    }

    Mat dst; 
    equalizeHist(src, dst); // 이 함수는 gray scale로 변환된것 8UC1만 가능. 
    
    imshow("src", src);
    imshow("srcHist", getGrayHistImage(calcGrayHist(src)));

    imshow("dst", dst);
    imshow("dstHist", getGrayHistImage(calcGrayHist(dst)));

    waitKey();
    destroyAllWindows();

}

Mat getGrayHistImage(const Mat& hist) {
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.size() == Size(1, 256));

    double histMax; 
    minMaxLoc(hist, 0, &histMax); 

    Mat imgHist(100, 256, CV_8UC1, Scalar(255));

    for (int i = 0; i < 256; i++) {
        line(imgHist, Point(i, 100), Point(i, 100 - cvRound(hist.at<float>(i, 0) * 100 / histMax)), Scalar(0));
    }
    return imgHist; 
}

Mat calcGrayHist(const Mat& img) {
    CV_Assert(img.type() == CV_8UC1);

    Mat hist;
    int channels[] = { 0 };
    int dims = 1;
    const int histSize[] = { 256 };
    float graylevel[] = { 0,256 };
    const float* ranges[] = { graylevel };

    calcHist(&img, 1, channels, Mat(), hist, dims, histSize, ranges);
    return hist;
}

int filter_embossing(void) {
    Mat src = imread("images/rose.bmp",IMREAD_GRAYSCALE);
     
    if (src.empty()) {
        cerr << "Image load failed" << endl;
        return -1;
    }

    float data[] = { -1,-1,0,-1,0,1,0,1,1 }; 
    Mat emboss(3, 3, CV_32FC1, data); 

    Mat dst, dst_delta0; 
    filter2D(src, dst, -1, emboss, Point(-1, -1), 128); 
    filter2D(src, dst_delta0, -1, emboss, Point(-1, -1));

    imshow("src", src); 
    imshow("dst", dst);
    imshow("dst-delta : 0", dst_delta0);

    waitKey();
    destroyAllWindows(); 
    return 0; 
}

int color_backprojection(void) {

    Mat ref, ref_ycrcb, mask;
    ref = imread("images/ref.png", IMREAD_COLOR);
    mask = imread("images/mask.bmp", IMREAD_GRAYSCALE);
    cvtColor(ref, ref_ycrcb, COLOR_BGR2YCrCb);

    //ref 사진에서 피부색 영역의 CrCb 2차원 히스토그램을 계산하여 hist에 저장. 
    Mat hist;
    int channels[] = { 1,2 }; // cr:1 cb:2 [y(명도):0은 제외]
    int cr_bins = 128, cb_bins = 128; // 128 = 255/2 (0번 bin은 픽셀값이 0또는 1)
    int histSize[] = { cr_bins, cb_bins }; //각 차원의 히스토그램 배열 크기
    float cr_range[] = { 0,256 }; //cr 차원의 히스토그램 범위 
    float cb_range[] = { 0, 256 };//cb 차원의 히스토그램 범위 
    const float* ranges[] = { cr_range, cb_range };

    //입력 영상 ref_ycrcb에서 히스토그램 hist를 따르는 픽셀을 찾고, 
    // 그 정보를 backProject 영상으로 반환하기 

    calcHist(&ref_ycrcb, 1, channels, mask, hist, 2, histSize, ranges);
    //calchist($ref_ycrcb: 입력영상 주소, 1: 입력영상 갯수, 
    // channels: 히스토그램을 구할 채널을 나타내는 정수형 배열 , mask: 마스크 영상 (입력 영상과 크기가 같은 8비트 배열)
    //hist:  출력 히스토그램 , 2(dims):  출력 히스토그램의 차원 수 
    //histSize: 각차원의 히스토그램 배열 크기를 나타내는 배열 (각 차원의 히스토그램 빈 갯수를 나타내는 배열) 
    //ranges: 각 차원의 히스토그램 범위 

    //histogram CrCb 확인하기 
    double maxVal = 0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    cout << "=========" << maxVal << endl;
    int scale = 5;
    Mat histImg = Mat::zeros(cr_bins * scale, cb_bins * scale, CV_8UC3);
    Mat histImg2 = Mat::zeros(cr_bins * scale, cb_bins * scale, CV_8UC3);
    for (int cr = 0; cr < cr_bins; cr++) {
        for (int cb = 0; cb < cb_bins; cb++)
        {
            float binVal = hist.at<float>(cr, cb);
            //밀도의 범위 (0~255)  
            int intensity = cvRound(binVal * 255 / maxVal);
            rectangle(histImg, Point(cb * scale, cr * scale),
                Point((cb + 1) * scale - 1, (cr + 1) * scale - 1),
                Scalar::all(intensity),-1);
        }
    }
    for (int cr = 0; cr < cr_bins; cr++) {
        for (int cb = 0; cb < cb_bins; cb++)
        {
            float binVal = hist.at<float>(cr, cb);
            //밀도의 범위 (0~255)  
            int intensity = cvRound(binVal * 255 / maxVal);
            rectangle(histImg2, Point(cb * scale, cr * scale),
                Point((cb * scale)+scale-1, (cr * scale)+scale-1),
                Scalar::all(intensity), -1);
        }
    }
    Mat dst;
    subtract(histImg, histImg2, dst);
    imshow("CrCb Histogram", histImg);
    imshow("CrCb Histogram22", histImg2);
    imshow("subtract", dst);

    waitKey();
    destroyAllWindows();

    Mat src, src_ycrcb, src2, src2_ycrcb;
    src = imread("images/kids.png", IMREAD_COLOR);
    src2 = imread("images/kids2.jpg");
    cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb); 
    cvtColor(src2, src2_ycrcb, COLOR_BGR2YCrCb);


    Mat backproj, backproj2;
    calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);
    calcBackProject(&src2_ycrcb, 1, channels, hist, backproj2, ranges, 1, true);

    // 1: 히스토그램 역투영 값에 추가적으로 곱할 값 
    // true: 히스토그램 빈의 간격이 균등하다 
    
    imshow("mask", mask);
    imshow("src", src);
    imshow("backproj", backproj);
    imshow("backproj2", backproj2);


    if (waitKey() == 27) {
        destroyAllWindows();
        return 0;
    }
    
}


Mat src, src_hsv, mask, bg;
int lower_hue = 40, upper_hue = 80;
int color_inrange(void) {
    //in Range()함수를 이용한 특정 색상 분할 
    // Hue: 색상 Saturation:채도 Value: 명도 
    
    src = imread("images/color_w.png",IMREAD_COLOR);
    bg = src.clone(); //Mat bg(src.size(), src.type());
    bg.setTo(Scalar(255, 255, 255));

    cvtColor(src, src_hsv, COLOR_BGR2HSV);
    imshow("src", src);

    namedWindow("mask");
    createTrackbar("Lower Hue", "mask", &lower_hue, 179, on_hue_changed);
    createTrackbar("Upper Hue", "mask", &upper_hue, 179, on_hue_changed);
    on_hue_changed(0, 0); //프로그램이 처음 실행될때, 영상이 정상 출력되도록 트랙바 콜백 함수를 강제로 호출 

    if(waitKey() == 27) {
        destroyAllWindows();
        return 0;
    }

}

void on_hue_changed(int, void*) {
    // S(채도)의 범위는 100~255로 임의 지정, 
    // V(명도)의 영향은 무시하도록 0~255 지정, 
    Scalar lowerb(lower_hue, 0, 0); //(h,s,v)
    Scalar upperb(upper_hue, 255, 255);

    //이거 없이 하려면, 원하는 출력이 안나옴 
    //아래와 같이 항상 깨끗이 mask를 정리해야함. 
    //mask = Mat::zeros(src.size(), CV_8UC1);
    inRange(src_hsv, lowerb, upperb, mask);
    imshow("mask", mask);
    //Mat mask_inv = Mat::zeros(src.size(), CV_8UC1);
    //bitwise_not(mask, mask_inv);

    //Mat img1 = Mat::zeros(src.size(), CV_8UC3);
    //bitwise_and(src, src, img1, mask);

    //Mat img2 = Mat::zeros(src.size(), CV_8UC3);
    //bitwise_and(bg, bg, img2, mask_inv); // mask = mask_inv

    //Mat final = Mat::zeros(src.size(), CV_8UC3);
    //add(img1, img2, final);
    //imshow("mask", final);

    
}

void nothing(int, void*) {
    return;
}

int color_inrange_v2(void) {
    Mat src = imread("images/color_w.png", IMREAD_COLOR);
    
    Mat bg = src.clone(); //Mat bg(src.size(), src.type());
    bg.setTo(Scalar(255, 255, 255));

    Mat src_hsv;
    cvtColor(src, src_hsv, COLOR_BGR2HSV);
    imshow("src", src);

    Mat channels[3]; //h,s,v
    split(src_hsv, channels);
    
    namedWindow("mask");
    int lower_hue=10, upper_hue=80;
    createTrackbar("Lower Hue", "mask", &lower_hue, 179, nothing);
    createTrackbar("Upper Hue", "mask", &upper_hue, 179, nothing);

    Mat mask, final, mask_inv, img1, img2;
    while (true) {
        lower_hue = getTrackbarPos("Lower Hue", "mask");
        upper_hue = getTrackbarPos("Upper Hue", "mask");

        mask = Mat::zeros(src.size(), CV_8UC1);
        inRange(channels[0], lower_hue, upper_hue, mask); //lo up 사이에 있으면 흰색, 않으면 검은색
        
        mask_inv = Mat::zeros(src.size(), CV_8UC1);
        bitwise_not(mask, mask_inv);
        
        img1 = Mat::zeros(src.size(), CV_8UC3);
        bitwise_and(src, src, img1,mask);

        img2 = Mat::zeros(src.size(), CV_8UC3);
        bitwise_and(bg, bg, img2,mask_inv); // mask = mask_inv

        final = Mat::zeros(src.size(), CV_8UC3);
        add(img1, img2, final);
        imshow("mask", final);
        if (waitKey(1) == 27) {
            destroyAllWindows();
            return 0;

        }
    }

}

int color_equalization(void) {
    Mat src = imread("images/lena.jpg", IMREAD_COLOR);
    vector<string> file_list = { "./images/airplane1.jpg", "./images/house.jpg",
        "./images/baboon.jpg", "./images/flower2.jpg",
        "./images/red_sky.jpg", "./images/puppies.jpg" };

    if (src.empty()) {
        cerr << "Image load failed" << endl;
        return -1;
    }

    //FROM BGR TO YCRCB 
    Mat src_ycrcb;
    cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb);
    
    //Split
    vector<Mat> ycrcb_planes;
    split(src_ycrcb, ycrcb_planes); // [0]: y, [1]:cr, [2]:cb

    //히스토그램 평활화 
    equalizeHist(ycrcb_planes[0], ycrcb_planes[0]); // src, dst

    //Merge
    Mat dst_ycrcb; 
    merge(ycrcb_planes, dst_ycrcb); //list merge! pixel value = [y,cr,cb]
    
    //FROM YCRCB TO BGR 
    Mat dst; //bgr 형태로 받을 mat 변수 
    cvtColor(dst_ycrcb, dst, COLOR_YCrCb2BGR);

    imshow("src", src);
    imshow("dst", dst);

    cout << "Press ESC key to close this window " << endl;
    if (waitKey() == 27) {
        destroyAllWindows();
        return 0;
    }

}


int printCodeList(void) {

    cout << "==========Print Code List=========" << endl;

    FILE* stream;
    //함수: 코드 번호 
    if (fopen_s(&stream, FILENAME, "r+") != 0) {//성공시 0 
        printf("The file was not opened\n");
        return -1;
    }

    int codeNum = 0;
    int pages = 0;
    char explain[100];

    // Read data back from file:
    fgets(explain, 100, stream); //throw headline
    printf("%s", explain);
    //memset(explain, 0, sizeof(char) * 100);

    while (fscanf(stream, "%d", &codeNum) > 0) {
        fscanf(stream, "%d", &pages);
        fgets(explain, 100, stream);
        // Output data read:
        printf("%d\t:", codeNum);
        printf("%d\t", pages);
        printf("%s", explain);
    }
    fclose(stream);

    return 0;

}
