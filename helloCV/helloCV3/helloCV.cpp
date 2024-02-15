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
int affine_rotation(void);
int flip(void);
void on_mouse(int event, int x, int y, int flags, void*);

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
            case 85: affine_rotation(); break; 
            case 86: flip(); break; 
            case 104: color_equalization(); break;
            case 105: color_inrange(); break;
            case 1052: color_inrange_v2(); break;
            case 106: color_backprojection(); break;
            
            case 0: cout << "Program is closed ..." << endl; return 0;
            default: cout << "Wrong Code ID, Retry!!!!!!!!" << endl; break;
        }
    
    }
    
}

int flip(void) {//reflection 좌우반전 >0 , 상하반전 = 0 , 좌우상하반전 < 0

    Mat src = imread("images/rose.bmp");
    if (src.empty()) {
        cerr << "src not unload" << endl;
        return -1;
    }
    Mat lr_dst, ud_dst, lrud_dst;
    flip(src, lr_dst, 1);
    flip(src, ud_dst, 0);
    flip(src, lrud_dst, -1);

    imshow("src",src);
    imshow("lr_dst", lr_dst);
    imshow("ud_dst", ud_dst);
    imshow("lrud_dst", lrud_dst);

    waitKey(); 
    destroyAllWindows();
    return 0; 
}

int affine_rotation() {
    Mat src = imread("images/rose.bmp");
    if (src.empty()) {
        cerr << "src not unload" << endl;
        return -1;
    }

    ///추가: 회전해도 전체 사진이 보이게
    int w = src.cols;
    int h = src.rows;

    cout << "몇도를 회전하실 것인지? ex) 반시계 +20, 시계-20";
    int angle = 0;
    cin >> angle;

    double rad = (angle * CV_PI) / 180;
    double cos_value = cos(rad);
    double sin_value = sin(rad);

    //회전 후 생성되는 영상의 크기를 계산
    // 4개의 코너 포인트의 이동 좌표를 계산하여 최대 최소점의 차이를 구한다. 
    // (0,0) (w,0) (0,h) (w,h) 

    int nx, ny, minx, miny, maxx, maxy, nw, nh;

    // (0,0) 
    minx = maxx = 0;
    miny = maxy = 0;

    //(w,0) 
    nx = static_cast<int>(floor((w - 1) * cos_value + 0.5));
    ny = static_cast<int>(-floor((w - 1) * sin_value + 0.5));
    minx = (minx < nx) ? minx : nx;
    maxx = (maxx > nx) ? maxx : nx;
    miny = (miny < ny) ? miny : ny;
    maxy = (maxy > ny) ? maxy : ny;

    //(0,h) 
    nx = static_cast<int>(floor((h - 1) * sin_value + 0.5));
    ny = static_cast<int>(floor((h - 1) * cos_value + 0.5));
    minx = (minx < nx) ? minx : nx;
    maxx = (maxx > nx) ? maxx : nx;
    miny = (miny < ny) ? miny : ny;
    maxy = (maxy > ny) ? maxy : ny;

    //(w,h) 
    nx = static_cast<int>(floor((w - 1) * cos_value + (h - 1) * sin_value));
    ny = static_cast<int>(floor(-(w - 1) * sin_value + (h - 1) * cos_value));
    minx = (minx < nx) ? minx : nx;
    maxx = (maxx > nx) ? maxx : nx;
    miny = (miny < ny) ? miny : ny;
    maxy = (maxy > ny) ? maxy : ny;

    nw = maxx - minx + 1;
    nh = maxy - miny + 1;
    Size newSZ(nw, nh);
    cout << "origin size (" << w << ":" << h << endl;
    cout << "new size (" << newSZ.width << ":" << newSZ.height << endl;
    string info = format("info>> minx: %d maxx: %d miny:%d maxy: %d",minx,maxx, miny, maxy);
    cout << info << endl;

    //Point2f cp(src.cols / 2.f, src.rows / 2.f);
    Point2f cp(0.f, 0.f);
    Mat M = getRotationMatrix2D(cp, angle, 1);
    cout << "m은 :\n" << M << endl;
    /*   origin size(480:320
       new size(560:465
           info >> minx : 0 maxx : 559 miny : -164 maxy : 300
           m은 :
           [0.9396926207859084, 0.3420201433256687, 0;
   -0.3420201433256687, 0.9396926207859084, 0]*/
    if (angle > 0) {
        double y_tr = -miny + 1.f;
        M.at<double>(1, 2) += y_tr;
        //Mat y_ = Mat_<double>( {2,3},{0,0,0,0,0,y_tr} );
        //M = M + y_; 
    }
    else { // 시계방향 
        double x_tr = -minx + 1.f;
        M.at<double>(0, 2) += x_tr;
    }
    cout << "이동후 m은 :\n" << M << endl;

    Mat dst;
    warpAffine(src, dst, M, newSZ);
    imshow("src", src);
    imshow("dst", dst);

    
    waitKey(); 
    destroyAllWindows(); 
}

int affine_scale() {
    Mat src = imread("images/rose.bmp");
    if (src.empty()) {
        cerr << "src not unload" << endl;
        return -1;
    }

    cout << "INTERPOLATION 선택하기" << endl;
    string interpolFlag[10] = { "INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC",
        "INTER_AREA", "INTER_LANCZOS4", "INTER_LINEAR_EXACT", "INTER_NEAREST_EXACT",
        "INTER_MAX", "WARP_FILL_OUTLIERS", "WARP_INVERSE_MAP" };

    for (int i = 0; i < 10; i++) {
        cout << i << ":" << interpolFlag[i] << endl;
    }
    int myinterpol = 1;
    cin >> myinterpol;

    Mat dst1, dst2, dst3, dst4,dst5,dst6; 
    resize(src, dst1, Size(), 4, 4, INTER_NEAREST); 
    resize(src, dst2, Size(1920, 1280)); //기본이 Bilinear interpolation 
    resize(src, dst3, Size(1920, 1280), 0, 0, INTER_NEAREST);
    resize(src, dst4, Size(1920, 1280), 0, 0, INTER_LINEAR);
    resize(src, dst5, Size(1920, 1280), 0, 0, INTER_CUBIC);
    resize(src, dst6, Size(1920, 1280), 0, 0, INTER_LANCZOS4);

    imshow("src", src); 
    imshow("dst1", dst1(Rect(400, 500, 400, 400))); 
    imshow("dst2", dst2(Rect(400, 500, 400, 400)));
    imshow("INTER_NEAREST", dst3(Rect(400, 500, 400, 400)));
    imshow("INTER_LINEAR", dst4(Rect(400, 500, 400, 400)));
    imshow("INTER_CUBIC", dst5(Rect(400, 500, 400, 400)));
    imshow("INTER_LANCZOS4", dst6(Rect(400, 500, 400, 400)));


    waitKey(); 
    destroyAllWindows();

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

    Mat dst;
    if (menu == 0) {
        cout << "가로로 얼만큼 밀겠습니까 (mx= 0.3)?";
        double mx;
        cin >> mx;
        Mat M = Mat_<double>({ 2,3 }, { 1,mx,0,0,1,0 });
        // x' = x+mx*y
        
        //BORDER_TRANSPARENT
        warpAffine(src, dst, M, Size(cvRound(src.cols + src.rows * mx), src.rows));
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

Mat src2;
Point ptOld;
vector<Point> markers;
vector<vector<Point>> markersGrp;
void on_mouse(int event, int x, int y, int flags, void*)
{
    switch (event) {
    case EVENT_LBUTTONDOWN:
        ptOld = Point(x, y);
        markers.push_back(ptOld);
        drawMarker(src2, ptOld, Scalar(0, 0, 0), MARKER_CROSS, 10);
        imshow("COLOR_PICKER", src2);
        cout << "point: " << x << ", " << y << endl;
        break;
    case EVENT_RBUTTONDOWN:
        markersGrp.push_back(markers);
        markers.clear();
        break;
    default:
        break;
    }
}


int color_backprojection(void) {
    vector<string> filename = { "images/ref.png","images/mask.bmp","images/kids.png"};//,"images/kids2.jpg", "images/kids3.jpg" 
    cout << "마스크를 뽑아낼 파일 경로를 입력하세요: 예) ""images/kids2.jpg"":" <<endl;
    string myfile;
    cin >> myfile;
    filename.push_back(myfile);
    cout << "만든 마스크로 테스트할 파일 경로를 입력하세요: 예) ""images/kids2.jpg"":" << endl;
    cin >> myfile;
    filename.push_back(myfile);
    cout << "이 함수에서 사용할 파일리스트 입니다 " << endl;
    for (int i = 0; i < filename.size(); i++) {
        cout << filename[i] << endl;
    }


    Mat ref, ref_ycrcb, mask;
    ref = imread(filename[0], IMREAD_COLOR);
    mask = imread(filename[1], IMREAD_GRAYSCALE);
    imshow("ref", ref);
    imshow("mask", mask); 
    waitKey();
    destroyAllWindows(); 

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
    cout << "hist size>>>" << hist.size() << endl;
    //calchist($ref_ycrcb: 입력영상 주소, 1: 입력영상 갯수, 
    // channels: 히스토그램을 구할 채널을 나타내는 정수형 배열 , mask: 마스크 영상 (입력 영상과 크기가 같은 8비트 배열)
    //hist:  출력 히스토그램 , 2(dims):  출력 히스토그램의 차원 수 
    //histSize: 각차원의 히스토그램 배열 크기를 나타내는 배열 (각 차원의 히스토그램 빈 갯수를 나타내는 배열) 
    //ranges: 각 차원의 히스토그램 범위 

    //histogram CrCb 확인하기 

    int scale = 5; //히스토그램 크기 확대 
    //Mat histImg = Mat::zeros(cr_bins * scale, cb_bins * scale, CV_8UC3);
    Mat histImg2 = Mat::zeros(cr_bins * scale, cb_bins * scale, CV_8UC1);
    //for (int cr = 0; cr < cr_bins; cr++) {
    //    for (int cb = 0; cb < cb_bins; cb++)
    //    {
    //        float binVal = hist.at<float>(cr, cb);
    //        //밀도의 범위 (0~255)  
    //        int intensity = cvRound(binVal * 255 / maxVal);
    //        rectangle(histImg, Point(cb * scale, cr * scale),
    //            Point((cb + 1) * scale - 1, (cr + 1) * scale - 1),
    //            Scalar::all(intensity),-1);
    //    }
    //}

    double maxVal = 0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);
    cout << "=========" << maxVal << endl;

    for (int cr = 0; cr < cr_bins; cr++) {
        for (int cb = 0; cb < cb_bins; cb++)
        {
            float binVal = hist.at<float>(cr, cb);
            //밀도의 범위 (0~255)  
            int intensity = cvRound(binVal * 255 / maxVal);
            rectangle(histImg2, Point(cb * scale, cr * scale),
                Point((cb * scale) + scale - 1, (cr * scale) + scale - 1),
                Scalar(intensity), -1);//Scalar::all(intensity)
        }
    }
    //Mat dst;
    /*subtract(histImg, histImg2, dst);
    imshow("CrCb Histogram", histImg);*/
    imshow("CrCb Histogram22", histImg2);
    //imshow("subtract", dst);

    waitKey();
    destroyAllWindows();

    Mat src, src_ycrcb, src2_ycrcb;
    src = imread(filename[2], IMREAD_COLOR);
    src2 = imread(filename[4]);
    cvtColor(src, src_ycrcb, COLOR_BGR2YCrCb); 
    cvtColor(src2, src2_ycrcb, COLOR_BGR2YCrCb);

    Mat backproj, backproj2;
    //&src_ycrcb : 입력영상, 입력영상개수, 채널번호배열, 입력히스토그램, 출력 역투영 영상, 
    // ranges: 각차원의 히스토그램 범위, 역투영 값에 추가적으로 곱할 값, 빈의 간격이 균등(uniform)하면 true 
    calcBackProject(&src_ycrcb, 1, channels, hist, backproj, ranges, 1, true);
    calcBackProject(&src2_ycrcb, 1, channels, hist, backproj2, ranges, 1, true);

    // 1: 히스토그램 역투영 값에 추가적으로 곱할 값 
    // true: 히스토그램 빈의 간격이 균등하다 
    
    imshow("src1", src);
    imshow("src2", src2);

    imshow("backproj1", backproj);
    imshow("backproj2", backproj2);

    waitKey();
    destroyAllWindows();

    //마스크 파일 만들기 
    namedWindow("COLOR_PICKER");
    setMouseCallback("COLOR_PICKER", on_mouse);

    src2 = imread(filename[3]);
    imshow("COLOR_PICKER", src2);
    waitKey();

    Mat src2_mask = Mat::zeros(src2.size(), CV_8UC1);
    for (int i = 0; i < markersGrp.size(); i++) {
        fillPoly(src2_mask, markersGrp[i], Scalar(255));
    }

    imshow("mask", src2_mask);
    waitKey(); 
    destroyAllWindows();

    cvtColor(src2, src2_ycrcb, COLOR_BGR2YCrCb);
    calcHist(&src2_ycrcb, 1, channels, src2_mask, hist, 2, histSize, ranges);
    //calcBackProject(&src2_ycrcb, 1, channels, hist, backproj2, ranges, 1, true);

    Mat src3 = imread(filename[4]);
    Mat src3_ycrcb, backproj3;
    cvtColor(src3, src3_ycrcb, COLOR_BGR2YCrCb);
    calcBackProject(&src3_ycrcb, 1, channels, hist, backproj3, ranges, 1, true);

    imshow("src3", src3);
    imshow("backproj3", backproj3);
    waitKey();
    destroyAllWindows();

    markers.clear();//초기화
    markersGrp.clear();

    return 0;
    
}



Mat src, src_hsv, mask, bg;
int lower_hue = 0, upper_hue = 0;
int color_inrange(void) {
    //in Range()함수를 이용한 특정 색상 분할 
    // Hue: 색상 Saturation:채도(0~255) Value: 명도(0~255) 
    // Hue: 0~179 (실제  0~360 이지만 256 UCHAR 로 표현하기 위해 /2를 함 ) 
    vector<string> file_list= {"images/candies.png","images/color_w.png"};
    
    for (int i = 0; i < file_list.size(); i++) {
        src = imread(file_list[i], IMREAD_COLOR);

        cvtColor(src, src_hsv, COLOR_BGR2HSV);
        imshow("src", src);

        namedWindow("mask", WINDOW_NORMAL);
        createTrackbar("Lower Hue", "mask", &lower_hue, 179, on_hue_changed);
        createTrackbar("Upper Hue", "mask", &upper_hue, 179, on_hue_changed);
        on_hue_changed(0, 0);
       
        int keyNum = waitKey();
        destroyAllWindows();

        if (keyNum == 27) {
            break;
        }
    }

    destroyAllWindows();
    return 0;
    

}

void on_hue_changed(int, void*) {
    // S(채도)의 범위는 100~255로 임의 지정, 
    // V(명도)의 영향은 무시하도록 0~255 지정, 
    Scalar lowerb(lower_hue, 0, 0); //(h,s,v)
    Scalar upperb(upper_hue, 255, 255);

    //mask = Mat::zeros(src.size(), CV_8UC1);    
    inRange(src_hsv, lowerb, upperb, mask);
    Mat bg = Mat::ones(src.size(), src.type());
    src.copyTo(bg, mask);
    imshow("mask", bg);
   
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
    int lower_hue=0, upper_hue=0;
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
    
    vector<string> file_list = { "images/pepper.bmp","images/lena.jpg","./images/airplane1.jpg", "./images/house.jpg",
        "./images/baboon.jpg", "./images/flower2.jpg",
        "./images/red_sky.jpg"};

    for (int i = 0; i < file_list.size(); i++) {

        TickMeter tm;
        tm.start();
        Mat src = imread(file_list[i], IMREAD_COLOR);

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

        
        imshow("dst", dst);
        tm.stop();
        cout << "<<time>>>>>>>>>>>>>>>>>> " << tm.getTimeMilli() << "ms" << endl;
        imshow("src", src);

        cout << "Press ESC key to close this window " << endl;
        int keyNum = waitKey();
        if (keyNum == 27) {
            break;
        }
        
    }

    destroyAllWindows();
    return 0;
    

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
