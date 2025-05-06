#include <iostream>
#include "asrl/vision/gpusurf/GpuSurfDetector.hpp"
#include "asrl/vision/gpusurf/GpuSurfStereoDetector.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>




using namespace std;
using namespace asrl;

void onTrackbar(int, void*) {
    // This callback does nothing
}


/// @brief convert ASLR keypoint type to opencv Keypoint.
cv::KeyPoint convertToCvKeyPoint(const Keypoint& kp) {
    return cv::KeyPoint(
        cv::Point2f(kp.x, kp.y),
        kp.size,
        kp.angle,
        kp.response,
        static_cast<int>(kp.octave)
    );
}

/// @brief converts integer list of mathcing features to opencv DMatch
std::vector<cv::DMatch> convertToDMatches(const std::vector<int>& matches) {
    std::vector<cv::DMatch> dmatches;
    dmatches.reserve(matches.size());
    for (size_t i = 0; i < matches.size(); ++i) {
        int rightIdx = matches[i];
        int leftIdx = static_cast<int>(i);

        if (rightIdx < 0) continue; // skip invalid matches if needed

        cv::DMatch dmatch(leftIdx, rightIdx, 0, 0.0f); 
        dmatches.push_back(dmatch);
    }
    return dmatches;
}

int main(int argc, char* argv[]) 
{
    //Set up the trackbar
    cv::namedWindow("GpuSurf", cv::WINDOW_NORMAL);
    cv::createTrackbar("Threshold/10000", "GpuSurf", nullptr, 10000, onTrackbar);
    cv::createTrackbar("nOctaves", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("nIntervals", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("initialScale", "GpuSurf", nullptr, 8, onTrackbar);
    cv::createTrackbar("edgeScale/10", "GpuSurf", nullptr, 1000, onTrackbar);
    cv::createTrackbar("targetFeatures", "GpuSurf", nullptr, 10000, onTrackbar);
    cv::createTrackbar("regionsVertical", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("regionsHorizontal", "GpuSurf", nullptr, 32, onTrackbar);
    cv::createTrackbar("regionsTarget", "GpuSurf", nullptr, 10000, onTrackbar);

    cv::createTrackbar("stereoDisparityMinimum/10", "GpuSurf", nullptr, 1, onTrackbar);
    cv::createTrackbar("stereoDisparityMaximum", "GpuSurf", nullptr, 500, onTrackbar);
    cv::createTrackbar("stereoCorrelationThreshold/10", "GpuSurf", nullptr, 10, onTrackbar);
    cv::createTrackbar("stereoYTolerance/10", "GpuSurf", nullptr, 10, onTrackbar);
    cv::createTrackbar("stereoScaleTolerance/10", "GpuSurf", nullptr, 10 , onTrackbar);

    cv::setTrackbarPos("Threshold/10000", "GpuSurf", 1);
    cv::setTrackbarPos("nOctaves", "GpuSurf", 4);
    cv::setTrackbarPos("nIntervals", "GpuSurf", 4);
    cv::setTrackbarPos("initialScale", "GpuSurf", 2);
    cv::setTrackbarPos("edgeScale/10", "GpuSurf", 100);
    cv::setTrackbarPos("targetFeatures", "GpuSurf", 1000);
    cv::setTrackbarPos("regionsVertical", "GpuSurf", 1);
    cv::setTrackbarPos("regionsHorizontal", "GpuSurf", 1);
    cv::setTrackbarPos("regionsTarget", "GpuSurf", 8192);

    cv::setTrackbarPos("stereoDisparityMinimum/10", "GpuSurf", 0);
    cv::setTrackbarPos("stereoDisparityMaximum", "GpuSurf", 120);
    cv::setTrackbarPos("stereoCorrelationThreshold/10", "GpuSurf", 2);
    cv::setTrackbarPos("stereoYTolerance/10", "GpuSurf", 9);
    cv::setTrackbarPos("stereoScaleTolerance/10", "GpuSurf", 1);


    cout << "Hello World! \n";
    cv::Mat img1 = cv::imread("/home/adam/Desktop/new_left/image_100.png", 0);
    cv::Mat img2 = cv::imread("/home/adam/Desktop/new_right/image_100.png", 0);
    if(img1.empty())
    {
        std::cout << "Could not read the image " << std::endl;
        return 1;
    }

    while (true) {
        float thresh = cv::getTrackbarPos("Threshold/10000", "GpuSurf")/10.0f;
        int nOctaves = cv::getTrackbarPos("nOctaves", "GpuSurf");
        int nIntervals = cv::getTrackbarPos("nIntervals", "GpuSurf");
        float initialScale = cv::getTrackbarPos("initialScale", "GpuSurf");
        float edgeScale = cv::getTrackbarPos("edgeScale/10", "GpuSurf") / 10.0f;
        int targetFeatures = cv::getTrackbarPos("targetFeatures", "GpuSurf");
        int regionsVertical = cv::getTrackbarPos("regionsVertical", "GpuSurf");
        int regionsHorizontal = cv::getTrackbarPos("regionsHorizontal", "GpuSurf");
        int regionsTarget = cv::getTrackbarPos("regionsTarget", "GpuSurf");
        float stereoDisparityMinimum = cv::getTrackbarPos("stereoDisparityMinimum/10", "GpuSurf")/10.0f;
        float stereoDisparityMaximum = cv::getTrackbarPos("stereoDisparityMaximum", "GpuSurf");
        float stereoCorrelationThreshold = cv::getTrackbarPos("stereoCorrelationThreshold/10", "GpuSurf")/10.0f;
        float stereoYTolerance = cv::getTrackbarPos("stereoYTolerance/10", "GpuSurf")/10.0f;
        float stereoScaleTolerance = cv::getTrackbarPos("stereoScaleTolerance/10", "GpuSurf")/10.0f;


        // stereo matching parameters
        GpuSurfStereoConfiguration configStereo;
        configStereo.stereoDisparityMinimum=stereoDisparityMinimum;
        configStereo.stereoDisparityMaximum=stereoDisparityMaximum;
        configStereo.stereoCorrelationThreshold=stereoCorrelationThreshold;
        configStereo.stereoYTolerance=stereoYTolerance;
        configStereo.stereoScaleTolerance=stereoScaleTolerance;
        configStereo.threshold = thresh;
        configStereo.nOctaves = nOctaves;
        configStereo.nIntervals = nIntervals;
        configStereo.initialScale = initialScale;
        configStereo.l1 = 3.f/1.5f;
        configStereo.l2 = 5.f/1.5f;
        configStereo.l3 = 3.f/1.5f;
        configStereo.l4 = 1.f/1.5f;
        configStereo.edgeScale = edgeScale;
        configStereo.initialStep = 1;
        configStereo.targetFeatures = targetFeatures;
        configStereo.detector_threads_x = 16;
        configStereo.detector_threads_y = 4;
        configStereo.nonmax_threads_x = 16;
        configStereo.nonmax_threads_y = 16;
        configStereo.regions_horizontal = regionsHorizontal;
        configStereo.regions_vertical = regionsVertical;
        configStereo.regions_target = regionsTarget;


        GpuSurfStereoDetector detector(configStereo);
        detector.setImages(img1, img2);
        detector.detectKeypoints();
        detector.findOrientation();

        detector.computeDescriptors(false); //weighted flag =false
        detector.matchKeypoints();

        //Retrieve the keypoints from the GPU.
        //vector<cv::KeyPoint> keypoints;
        vector<vector<asrl::Keypoint>> keypoints;
        keypoints.resize(2);
        vector<int> leftRightMatches;
        detector.getKeypoints(keypoints[0], keypoints[1], leftRightMatches);
        //get descriptors
        vector<float> descriptors;
        detector.getDescriptors(descriptors);
        int descriptor_size = detector.descriptorSize();
    
        
        // // I can see that we get about 64 descriptors for each keypoint
        std::cout << "Keypoints left: " << keypoints[0].size() << "   Keypoints right: "  << keypoints[1].size()  << std::endl;

        
        int count=0;
        vector<float> matches;
        for (size_t i = 0; i < leftRightMatches.size(); ++i) {
            if (leftRightMatches[i] >= 0) {
                count++;
                matches.push_back(leftRightMatches[i]);
            }
        }
        std::cout << "matches " << count << std::endl;
        
        

        //convert asrl keypoints to cv::KeyPoint
        vector<cv::KeyPoint> keypoints1, keypoints2;
        for (const auto& kp : keypoints[0]) {
            keypoints1.push_back(convertToCvKeyPoint(kp));
        }
        for (const auto& kp : keypoints[1]) {
            keypoints2.push_back(convertToCvKeyPoint(kp));
        }

        vector<cv::DMatch> best_matches=convertToDMatches(leftRightMatches);    
        //vector<cv::DMatch> best_matches(matches.begin(), matches.begin() + std::min(matches.size(), size_t(100)));

        // // drawing the results
        cv::namedWindow("matches", 1);
        cv::Mat img_matches;
        cv::drawMatches(img1, keypoints1, img2, keypoints2, best_matches, img_matches);
        cv::putText(img_matches, "Matches: "+to_string(count),cv::Point(25, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        cv::imshow("matches", img_matches);

        if (cv::waitKey(30) == 27) {
            break;  // Exit on ESC key
        }
    }
    return 0;
}