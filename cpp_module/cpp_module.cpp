// cpp_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace py = pybind11;\
using namespace cv;

py::array_t<uint8_t> blur_largest_shape_in_rect(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15) {

    // Convert py::array to cv::Mat
    py::buffer_info buf = input_array.request();
    int height = buf.shape[0];
    int width = buf.shape[1];
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;

    Mat input_img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, buf.ptr);

    // Copy for output
    Mat output_img = input_img.clone();

    // Extract rect coordinates
    if (rect_tuple.size() != 4)
        throw std::runtime_error("rect_tuple must contain 4 elements (x, y, w, h)");

    int x = rect_tuple[0].cast<int>();
    int y = rect_tuple[1].cast<int>();
    int w = rect_tuple[2].cast<int>();
    int h = rect_tuple[3].cast<int>();

    Rect roi(x, y, w, h);
    if ((x + w > width) || (y + h > height))
        throw std::runtime_error("Rectangle out of image bounds");

    // Convert to grayscale for contour detection
    Mat gray;
    if (channels == 3)
        cvtColor(input_img, gray, COLOR_BGR2GRAY);
    else
        gray = input_img;  

    // Crop to region of interest
    Mat roi_img = gray(roi);

    Mat img_blur, img_canny;

    // Blurring image using gaussian fliter. Size(3,3) is SE kernal
    GaussianBlur(roi_img, img_blur, Size(3, 3), 0);

    // edge detection using Canny
    Canny(img_blur, img_canny, 10, 150);

    // running dilation to close gaps in edges
    Mat dilated;
    dilate(img_canny, dilated, Mat(), Point(-1, -1), 2);

    // Find contours
    std::vector<std::vector<Point>> contours;
    findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return input_array; // nothing to blur

    // Find largest contour by area   
    size_t largest_idx = 0;
    double max_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i)
    {
        double area = contourArea(contours[i]);

        if (area > max_area)
        {
            max_area = area;
            largest_idx = i;
        }
    }  

    // Create mask for the largest shape
    Mat mask = Mat::zeros(roi.size(), CV_8UC1);
    drawContours(mask, contours, static_cast<int>(largest_idx), Scalar(255), FILLED);

    // Blur the ROI and copy only the masked region to output
    Mat blurred_roi;
    GaussianBlur(input_img(roi), blurred_roi, Size(blur_kernel, blur_kernel), 0);

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
