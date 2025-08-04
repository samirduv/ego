// cpp_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace py = pybind11;\
using namespace cv;

// Function that evaluates the edges of the image to determine the most prominent color
// used to determine the background color of the image
uchar getMostDominantBorderColor(const cv::Mat& img) {
    std::unordered_map<uchar, int> colorCounter;

    for (int x = 0; x < img.cols; ++x) {
        colorCounter[img.at<uchar>(0, x)]++;
        colorCounter[img.at<uchar>(img.rows - 1, x)]++;
    }
    for (int y = 0; y < img.rows; ++y) {
        colorCounter[img.at<uchar>(y, 0)]++;
        colorCounter[img.at<uchar>(y, img.cols - 1)]++;
    }

    uchar dominantColor = 0;
    int maxCount = 0;
    for (const auto& entry : colorCounter) {
        uchar val = entry.first;
        int count = entry.second;
        if (count > maxCount) {
            dominantColor = val;
            maxCount = count;
        }
    }

    return dominantColor;
}

py::array_t<uint8_t> blur_largest_shape_in_rect(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15) {

    // Convert py::array to cv::Mat
    py::buffer_info buf = input_array.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;

    cv::Mat input_img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, buf.ptr);

    // Copy for output
    cv::Mat output_img = input_img.clone();

    // Extract rect coordinates
    if (rect_tuple.size() != 4)
        throw std::runtime_error("rect_tuple must contain 4 elements (x, y, w, h)");

    int x = rect_tuple[0].cast<int>();
    int y = rect_tuple[1].cast<int>();
    int w = rect_tuple[2].cast<int>();
    int h = rect_tuple[3].cast<int>();

    cv::Rect roi(x, y, w, h);
    if ((x + w > width) || (y + h > height))
        throw std::runtime_error("Rectangle out of image bounds");

    // Convert to grayscale for contour detection
    cv::Mat gray;
    if (channels == 3)
        cv::cvtColor(input_img, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input_img;  

    // Crop to region of interest
    cv::Mat roi_img = gray(roi);

    // // Threshold to binary
    cv::Mat binary;
    cv::threshold(roi_img, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    uchar borderColor = getMostDominantBorderColor(binary);

    // Invert binary image if the border color is white (255)
    if (borderColor == 255) {
        cv::bitwise_not(binary, binary);
    }

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return input_array; // nothing to blur

    // Find largest contour by area
    size_t largest_idx = 0;
    double max_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = cv::contourArea(contours[i]);

        // write for debugging
        // cv::Mat contour_img = cv::Mat::zeros(roi.size(), CV_8UC1);
        // cv::drawContours(contour_img, contours, static_cast<int>(i), cv::Scalar(255), cv::FILLED);
        // std::string contour_filename = "contour_" + std::to_string(i) + ".png";
        // cv::imwrite(contour_filename, contour_img);

        if (area > max_area)
        {
            max_area = area;
            largest_idx = i;
        }
    }     

    // Create mask for the largest shape
    cv::Mat mask = cv::Mat::zeros(roi.size(), CV_8UC1);
    cv::drawContours(mask, contours, static_cast<int>(largest_idx), cv::Scalar(255), cv::FILLED);

    // write for debugging
    cv::imwrite("mask.png", mask);

    // Blur the ROI and copy only the masked region to output
    cv::Mat blurred_roi;
    cv::GaussianBlur(input_img(roi), blurred_roi, cv::Size(blur_kernel, blur_kernel), 0);

    // Apply mask to selectively blur
    blurred_roi.copyTo(output_img(roi), mask);

    // Return result as numpy array
    py::array_t<uint8_t> result = py::array_t<uint8_t>(buf.shape);
    py::buffer_info result_buf = result.request();
    std::memcpy(result_buf.ptr, output_img.data, buf.size * sizeof(uint8_t));
    return result;
}

// Module registration – candidate does not need to change this
PYBIND11_MODULE(cpp_module_edge_detector, m)
{
    m.def(
        "blur_largest_shape_in_rect", &blur_largest_shape_in_rect,
        py::arg("input_array"),
        py::arg("rect_tuple"),
        py::arg("blur_kernel") = 15,
        "Detect shapes in a rectangle and blur the largest shape.");
}
