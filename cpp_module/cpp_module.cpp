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

    // --- Edge pipeline (ROI only) ---
    Mat blurred, edges, dilated;
    GaussianBlur(gray, blurred, Size(3, 3), 0);
    Canny(blurred, edges, 40, 80);
    dilate(edges, dilated, Mat(), Point(-1, -1), 2); // close gaps

    // --- Contours in ROI ---
    std::vector<std::vector<Point>> contours;
    findContours(dilated, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return input_array; // nothing to blur

    // --- Largest contour by area ---
    size_t largest_idx = 0;
    double max_area = 0.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = contourArea(contours[i]);
        if (a > max_area) { max_area = a; largest_idx = i; }
    }

    // --- Approx to polygon ---
    const auto &c = contours[largest_idx];
    double peri = arcLength(c, true);
    std::vector<Point> poly;
    approxPolyDP(c, poly, 0.02 * peri, true); // tune 0.01–0.03

    if (!poly.empty()) {
        // Draw red if BGR, else white if single-channel
        Scalar color = (channels == 3) ? Scalar(0, 0, 255) : Scalar(255);

        // polylines expects a vector<vector<Point>>; draw on ROI of output
        polylines(output_img(roi),
                  std::vector<std::vector<Point>>{poly},
                  true, color, 2, LINE_AA);
    }

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
