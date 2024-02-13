// helloCV.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "opencv2/opencv.hpp"
#include <iostream>

#define FILENAME "test.txt"

using namespace cv;
using namespace std;

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
int noise(void);
int filter_bilateral(void);
int filter_median(void);
int blurring_mean(void); 
int blurring_gaussian(void); 
int sharpen(void); 
int affine(void); 
int affine_translation(void); 
int affine_shear(void); 
int affine_scale(void); 

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
            case 510: histogram_equalization(); break;
            case 71: filter_embossing(); break;
            case 72: blurring_mean(); break;
            case 73: blurring_gaussian(); break;
            case 74: sharpen(); break;       
            case 75: noise(); break;
            case 76: filter_bilateral(); break;
            case 77: filter_median(); break; 
            case 81: affine(); break;
            case 82: affine_translation(); break; 
            case 83: affine_shear(); break; 
            case 84: affine_scale(); break; 
            case 104: color_equalization(); break;
            case 105: color_inrange(); break;
            case 1052: color_inrange_v2(); break;
            case 106: color_backprojection(); break;
            
            case 0: cout << "Program is closed ..." << endl; return 0;
            default: cout << "Wrong Code ID, Retry!!!!!!!!" << endl; break;
        }
    
    }
    
}

int affine_scale() {
    Mat src = imread("images/rose.bmp");
    if (src.empty()) {
        cerr << "src not unload" << endl;
        return -1;
    }


    return 0; 
}

int affine_shear() {
    Mat src = imread("images/tekapo.bmp");
    if (src.empty()) {
        cerr << "src not unload" << endl;
        return -1;
    }
    int menu = 0; 
    cout << "가로로 민다면 0, 세로로 민다면 1"; 
    cin >> menu; 
    cout << "INTERPOLATION 선택하기" << endl;
    int interpolFlag[10] = { INTER_NEAREST, INTER_LINEAR, INTER_CUBIC, 
        INTER_AREA, INTER_LANCZOS4, INTER_LINEAR_EXACT, INTER_NEAREST_EXACT, 
        INTER_MAX, WARP_FILL_OUTLIERS, WARP_INVERSE_MAP };
    
    for (int i = 0; i < 10; i++) {
        cout << i << ":" <<interpolFlag[i]<< endl;
    }
    int myFlag = 1; 
    cin >> myFlag; 

    Mat dst;
    if (menu == 0) {
        cout << "가로로 얼만큼 밀겠습니까 (mx= 0.3)?";
        double mx;
        cin >> mx;
        Mat M = Mat_<double>({ 2,3 }, { 1,mx,0,0,1,0 });
        // x' = x+mx*y
        
        //BORDER_TRANSPARENT
        warpAffine(src, dst, M, Size(cvRound(src.cols + src.rows * mx), src.rows), myFlag, 0, Scalar(0,0,255));
    }
    else if (menu == 1) {
        cout << "세로로 얼만큼 밀겠습니까 (my= 0.3)?";
        double my;
        cin >> my;
        Mat M = Mat_<double>({ 2,3 }, { 1,0,0,my,1,0 });
        // y' = y+my*x

        
        warpAffine(src, dst, M, Size(src.cols, cvRound(src.rows + src.cols * my)));
    }
    else {
        cout << "메뉴 잘못 고르심"; 
        return -1; 
    }

    imshow("src", src); 
    imshow("dst", dst);
    waitKey();
    destroyAllWindows(); 
}

int affine_translation() {
    Mat src = imread("images/tekapo.bmp");
    if (src.empty()) {
        cerr << "src not unload" << endl;
        return -1;
    }
    cout << "src size: " << src.size() << endl;
    double a = 0, b = 0;
    cout << "가로와 세로로 몇 픽셀 이동(a b): ";
    cin >> a >> b;
    
    Mat M = Mat_<double>({ 2,3 }, { 1,0,a,0,1,b }); 

    Mat dst;
    warpAffine(src, dst, M, Size()); 

    imshow("src", src); 
    imshow("dst", dst);
    waitKey(); 
    destroyAllWindows(); 
    return 0; 
}

int affine(void) {
    Mat src = imread("images/tekapo.bmp");
    if (src.empty()) {
        cerr << "src not unload" << endl;
        return -1; 
    }
    cout << "src size: "<< src.size() << endl;

    Point2f srcPts[3], dstPts[3]; 
    srcPts[0] = Point2f(0, 0);
    srcPts[1] = Point2f(src.cols-1, 0);
    srcPts[2] = Point2f(src.cols-1, src.rows-1);

    dstPts[0] = Point2f(50, 50); 
    dstPts[1] = Point2f(src.cols - 100, 100); 
    dstPts[2] = Point2f(src.cols - 50, src.rows - 50);

    Mat M = getAffineTransform(srcPts, dstPts); 
    vector<Point2f> src_loc = { Point2f(100,20), Point2f(200,50) }; 
    vector<Point2f> dst_loc; 
    transform(src_loc, dst_loc, M);
    cout << "affine matrix"<< M;


    Mat dst;
    warpAffine(src, dst, M, Size()); 

    imshow("src", src); 
    imshow("dst", dst); 
    waitKey(); 
    destroyAllWindows();

}

int filter_median(void) {
    Mat src = imread("images/lena.jpg", IMREAD_GRAYSCALE);
    imshow("src", src);
    if (src.empty()) {
        cerr << "image load failed" << endl;
        return -1;
    }

    int num = (int)(src.total() * 0.1);//전체 픽셀의 10%에 대해서 노이즈 추가 
    for (int i = 0; i < num; i++) {
        int x = rand() % src.cols;//랜덤 위치 선정 
        int y = rand() % src.rows; 
        src.at<uchar>(y, x) = (i % 2) * 255; //i가 홀수면 255, i가 짝수면 0
    }

    Mat dst1;
    GaussianBlur(src, dst1, Size(), 1);

    Mat dst2; 
    medianBlur(src, dst2, 3); 
    imshow("dst1", dst1);
    imshow("dst2", dst2);
    waitKey();
    destroyAllWindows();
}

int filter_bilateral(void) {
    Mat src = imread("images/lena.jpg", IMREAD_GRAYSCALE);
    imshow("src", src);
    if (src.empty()) {
        cerr << "image load failed" << endl;
        return -1;
    }

    Mat noise(src.size(), CV_32SC1); 
    randn(noise, 0, 5); //return array, mean, stddev(표준편차)
    add(src, noise, src, Mat(), CV_8U); 
    imshow("noise", src);

    Mat dst1; 
    GaussianBlur(src, dst1, Size(), 5); // Size()는 원래 kernel size인데 , 표준편차에 의해서 자동생성가능, 5:stddev 
    imshow("gaussian blur", dst1); 

    Mat dst2;
    bilateralFilter(src, dst2, -1, 10, 5); // -1 sigmaSpace로부터 자동생성됨. 10: 색공간에서의 가우시안 표준 편차 5: 좌표 공간에서의 가우시안 표준편차 
    imshow("bilateralFilter", dst2); 

    waitKey(); 
    destroyAllWindows(); 
}

int noise(void) {
    Mat src = imread("images/lena.jpg", IMREAD_GRAYSCALE); 

    if (src.empty()) {
        cerr << "image load failed" << endl;
        return -1;
    }

    imshow("src", src);

    for (int stddev = 10; stddev <= 30; stddev += 10) {
        Mat noise(src.size(), CV_32SC1);
        randn(noise, 0, stddev);

        Mat dst;
        add(src, noise, dst, Mat(), CV_8U);

        String  desc = format("stddev=%d", stddev);
        imshow(desc, dst);

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
