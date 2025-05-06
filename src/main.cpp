#include <iostream>
#include "asrl/vision/gpusurf/GpuSurfDetector.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace asrl;

void onTrackbar(int, void*) {
    // This callback does nothing
}

int main(int argc, char* argv[]) 
{
    //Set up the trackbar
    cv::namedWindow("GpuSurf", cv::WINDOW_NORMAL);
    cv::createTrackbar("Threshold/10", "GpuSurf", nullptr, 10, onTrackbar);
    cv::createTrackbar("nOctaves", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("nIntervals", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("initialScale", "GpuSurf", nullptr, 8, onTrackbar);
    cv::createTrackbar("edgeScale/10", "GpuSurf", nullptr, 1000, onTrackbar);
    cv::createTrackbar("targetFeatures", "GpuSurf", nullptr, 10000, onTrackbar);
    cv::createTrackbar("regionsVertical", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("regionsHorizontal", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("regionsTarget", "GpuSurf", nullptr, 10000, onTrackbar);

    cv::setTrackbarPos("Threshold/10", "GpuSurf", 1);
    cv::setTrackbarPos("nOctaves", "GpuSurf", 4);
    cv::setTrackbarPos("nIntervals", "GpuSurf", 4);
    cv::setTrackbarPos("initialScale", "GpuSurf", 2);
    cv::setTrackbarPos("edgeScale/10", "GpuSurf", 100);
    cv::setTrackbarPos("targetFeatures", "GpuSurf", 1000);
    cv::setTrackbarPos("regionsVertical", "GpuSurf", 1);
    cv::setTrackbarPos("regionsHorizontal", "GpuSurf", 1);
    cv::setTrackbarPos("regionsTarget", "GpuSurf", 8192);

    cout << "Hello World! \n";
    cv::Mat img1 = cv::imread("/home/adam/Desktop/new_left/image_100.png", 0);
    cv::Mat img2 = cv::imread("/home/adam/Desktop/new_right/image_100.png", 0);
    if(img1.empty())
    {
        std::cout << "Could not read the image " << std::endl;
        return 1;
    }

    while (true) {
        float thresh = cv::getTrackbarPos("Threshold/10", "GpuSurf")/10.0f;
        int nOctaves = cv::getTrackbarPos("nOctaves", "GpuSurf");
        int nIntervals = cv::getTrackbarPos("nIntervals", "GpuSurf");
        float initialScale = cv::getTrackbarPos("initialScale", "GpuSurf");
        float edgeScale = cv::getTrackbarPos("edgeScale/10", "GpuSurf") / 10.0f;
        int targetFeatures = cv::getTrackbarPos("targetFeatures", "GpuSurf");
        int regionsVertical = cv::getTrackbarPos("regionsVertical", "GpuSurf");
        int regionsHorizontal = cv::getTrackbarPos("regionsHorizontal", "GpuSurf");
        int regionsTarget = cv::getTrackbarPos("regionsTarget", "GpuSurf");

        GpuSurfConfiguration config;
        config.threshold = thresh;
        config.nOctaves = nOctaves;
        config.nIntervals = nIntervals;
        config.initialScale = initialScale;
        config.l1 = 3.f/1.5f;
        config.l2 = 5.f/1.5f;
        config.l3 = 3.f/1.5f;
        config.l4 = 1.f/1.5f;
        config.edgeScale = edgeScale;
        config.initialStep = 1;
        config.targetFeatures = targetFeatures;
        config.detector_threads_x = 16;
        config.detector_threads_y = 4;
        config.nonmax_threads_x = 16;
        config.nonmax_threads_y = 16;
        config.regions_horizontal = regionsHorizontal;
        config.regions_vertical = regionsVertical;
        config.regions_target = regionsTarget;

        GpuSurfDetector detector1(config);
        detector1.buildIntegralImage(img1);
        detector1.detectKeypoints();
        detector1.findOrientation();

        //Retrieve the keypoints from the GPU.
        vector<cv::KeyPoint> keypoints1;
        detector1.getKeypoints(keypoints1);
        detector1.computeDescriptors(false);
        vector<float> descriptors1;
        detector1.getDescriptors(descriptors1);

        GpuSurfDetector detector2(config);
        detector2.buildIntegralImage(img2);
        detector2.detectKeypoints();
        detector2.findOrientation();
        //Retrieve the keypoints from the GPU.
        vector<cv::KeyPoint> keypoints2;
        detector2.getKeypoints(keypoints2);
        detector2.computeDescriptors(false);
        vector<float> descriptors2;
        detector2.getDescriptors(descriptors2);

        // I can see that we get about 64 descriptors for each keypoint
        std::cout << "Keypoints left: " << keypoints1.size() << "   Keypoints right: "  << keypoints2.size()  << std::endl;

        //convert the descriptors to cv::Mat
        int descriptor_length = 64;  // For SURF, typically 64 or 128

        // Check that the vector size is consistent
        assert(descriptors1.size() == keypoints1.size() * descriptor_length);

        // Create a Mat with one row per keypoint, and 32-bit float descriptors
        //cv::Mat descriptors_mat(keypoints1.size(), descriptor_length, CV_32F, descriptors2.data());

        cv::Mat desc1(keypoints1.size(), 64, CV_32F, descriptors1.data());
        cv::Mat desc2(keypoints2.size(), 64, CV_32F, descriptors2.data());

        // matching descriptors
        cv::BFMatcher matcher(cv::NORM_L2);
        vector<cv::DMatch> matches;
        matcher.match(desc1, desc2, matches);

        // filter mathces 
        std::sort(matches.begin(), matches.end(), [](const cv::DMatch& a, const cv::DMatch& b) {
                return a.distance < b.distance;
        });
        std::vector<cv::DMatch> best_matches(matches.begin(), matches.begin() + std::min(matches.size(), size_t(100)));

        // drawing the results
        cv::namedWindow("matches", 1);
        cv::Mat img_matches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, best_matches, img_matches);
        cv::imshow("matches", img_matches);
        // int k= cv::waitKey(0);
        if (cv::waitKey(30) == 27) {
            break;  // Exit on ESC key
        }
    }
    return 0;
}