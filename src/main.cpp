#include <iostream>
#include "asrl/vision/gpusurf/GpuSurfDetector.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace asrl;


int main(int argc, char* argv[]) 
{
    cout << "Hello World! \n";
    cv::Mat img1 = cv::imread("/home/adam/Desktop/new_left/image_100.png", 0);
    cv::Mat img2 = cv::imread("/home/adam/Desktop/new_right/image_100.png", 0);
    if(img1.empty())
    {
        std::cout << "Could not read the image " << std::endl;
        return 1;
    }
    cv::imshow("Display window", img1);
    int k = cv::waitKey(0); // Wait for a keystroke in the window

    GpuSurfConfiguration config;
     // Manually set ALL values
    config.threshold = 0.1f;
    config.nOctaves = 4;
    config.nIntervals = 4;
    config.initialScale = 2.f;
    config.l1 = 3.f/1.5f;
    config.l2 = 5.f/1.5f;
    config.l3 = 3.f/1.5f;
    config.l4 = 1.f/1.5f;
    config.edgeScale = 0.81f;
    config.initialStep = 1;
    config.targetFeatures = 1000;
    config.detector_threads_x = 16;
    config.detector_threads_y = 4;
    config.nonmax_threads_x = 16;
    config.nonmax_threads_y = 16;
    config.regions_horizontal = 1;
    config.regions_vertical = 1;
    config.regions_target = 8192;


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

    // I can see that we get about 64 descriptors for each keypoint
    std::cout << "Keypoints left: " << " (" << keypoints1.size() << " total)" << std::endl;
    std::cout << "Descriptors left: " << descriptors1.size() << std::endl;
    for (size_t i = 0; i < 3; ++i) {
        const auto& kp = keypoints1[i];
        std::cout << "  [" << i << "] pt=(" << kp.pt.x << ", " << kp.pt.y << ")"
                  << ", size=" << kp.size
                  << ", angle=" << kp.angle
                  << ", response=" << kp.response
                  << ", octave=" << kp.octave
                  << ", class_id=" << kp.class_id 
                  << ", descriptor=" << descriptors1[i]  << std::endl;
    }

    // visualize keypoints
    cv::Mat output1;
    cv::drawKeypoints(img1, keypoints1, output1, cv::Scalar(0, 0, 255) , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Left Image", output1);
    k = cv::waitKey(0);


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

    std::cout << "Keypoints right: " << " (" << keypoints2.size() << " total)" << std::endl;
    std::cout << "Descriptors right: " << descriptors2.size() << std::endl;

    cv::Mat output2;
    cv::drawKeypoints(img2, keypoints2, output2, cv::Scalar(255, 0, 0) , cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Right Image", output2);
    k = cv::waitKey(0);

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
    k= cv::waitKey(0);

    // debugging
    for (size_t i = 0; i < 10; ++i) {
    cout << "debugging: " << matches[i].distance << std::endl;
    }
    cout << "finished\n";
    return 0;
}