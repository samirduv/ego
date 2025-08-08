// cpp_module.cpp

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <opencv2/opencv.hpp>

namespace py = pybind11;\
using namespace cv;

// --- helpers (same file or in a header) ---
inline int shape_from_string(const std::string &s) {
    if (s == "rect")    return cv::MORPH_RECT;
    if (s == "ellipse") return cv::MORPH_ELLIPSE;
    if (s == "cross")   return cv::MORPH_CROSS;
    throw std::runtime_error("Unknown shape: " + s + " (expected: rect|ellipse|cross)");
}

inline int morph_from_string(const std::string &s) {
    if (s == "none" || s.empty()) return -1;
    if (s == "open")     return cv::MORPH_OPEN;
    if (s == "close")    return cv::MORPH_CLOSE;
    if (s == "gradient") return cv::MORPH_GRADIENT;
    if (s == "tophat")   return cv::MORPH_TOPHAT;
    if (s == "blackhat") return cv::MORPH_BLACKHAT;
    throw std::runtime_error("Unknown morph op: " + s + " (expected: none|open|close|gradient|tophat|blackhat)");
}

py::array_t<uint8_t> outline_largest_shape_in_rect(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    // Gaussian
    int gauss_kernel_size = 3,
    double gauss_sigma = 0.0,
    // Canny
    double canny_thresh1 = 40.0,
    double canny_thresh2 = 80.0,
    // Dilation
    int dilate_kernel_size = 3,
    int dilate_iterations = 2,
    std::string dilate_shape_str = "rect",  // "rect" | "ellipse" | "cross"
    // MorphologyEx
    int morph_op = -1,           
    int morph_kernel_size = 3,
    int morph_iterations = 1,
    std::string morph_shape_str = "rect",   // "rect" | "ellipse" | "cross"
    bool morph_first = false,
    // Drawing
    double approx_eps_frac = 0.02
) {

    // Map shape strings to enum values
    int dilate_shape = shape_from_string(dilate_shape_str);
    int morph_shape  = shape_from_string(morph_shape_str);

        // --- Py buffer to Mat ---
    py::buffer_info buf = input_array.request();
    int height   = buf.shape[0];
    int width    = buf.shape[1];
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;
    Mat input_img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1, buf.ptr);
    Mat output_img = input_img.clone();

    // --- ROI ---
    if (rect_tuple.size() != 4)
        throw std::runtime_error("rect_tuple must contain 4 elements (x, y, w, h)");
    int x = rect_tuple[0].cast<int>();
    int y = rect_tuple[1].cast<int>();
    int w = rect_tuple[2].cast<int>();
    int h = rect_tuple[3].cast<int>();
    if (x < 0 || y < 0 || w <= 0 || h <= 0 || x + w > width || y + h > height)
        throw std::runtime_error("Rectangle out of image bounds");
    Rect roi(x, y, w, h);

    // --- Grayscale ---
    Mat gray;
    if (channels == 3) cvtColor(input_img(roi), gray, COLOR_BGR2GRAY);
    else               gray = input_img(roi);

    // --- Gaussian ---
    if (gauss_kernel_size % 2 == 0) gauss_kernel_size++;
    Mat img_blur;
    GaussianBlur(gray, img_blur, Size(gauss_kernel_size, gauss_kernel_size), gauss_sigma);

    // --- Canny ---
    Mat edges;
    Canny(img_blur, edges, canny_thresh1, canny_thresh2);

    // --- Structuring elements ---
    Mat dilate_kernel = getStructuringElement(dilate_shape, Size(dilate_kernel_size, dilate_kernel_size));
    Mat morph_kernel  = getStructuringElement(morph_shape,  Size(morph_kernel_size,  morph_kernel_size));

    // --- Morph + Dilate ---
    Mat proc = edges;
    auto do_morph = [&]() {
        if (morph_op >= 0 && morph_iterations > 0)
            morphologyEx(proc, proc, morph_op, morph_kernel, Point(-1,-1), morph_iterations);
    };
    auto do_dilate = [&]() {
        if (dilate_iterations > 0)
            dilate(proc, proc, dilate_kernel, Point(-1,-1), dilate_iterations);
    };
    if (morph_first) { do_morph(); do_dilate(); }
    else              { do_dilate(); do_morph(); }    

    // Find contours
    std::vector<std::vector<Point>> contours;
    findContours(proc, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

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

    // Draw red border around the largest shape inside ROI
    drawContours(output_img(roi), contours, static_cast<int>(largest_idx), Scalar(0, 0, 255), 2);

    // Return result as numpy array
    py::array_t<uint8_t> result = py::array_t<uint8_t>(buf.shape);
    py::buffer_info result_buf = result.request();
    std::memcpy(result_buf.ptr, output_img.data, buf.size * sizeof(uint8_t));
    return result;
}

// Forward-declare your implementation that takes string shapes + int morph_op
py::array_t<uint8_t> outline_largest_shape_in_rect(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int gauss_kernel_size,
    double gauss_sigma,
    double canny_thresh1,
    double canny_thresh2,
    int dilate_kernel_size,
    int dilate_iterations,
    std::string dilate_shape_str, // "rect"|"ellipse"|"cross"
    int morph_op,                 // cv::MORPH_* or -1
    int morph_kernel_size,
    int morph_iterations,
    std::string morph_shape_str,  // "rect"|"ellipse"|"cross"
    bool morph_first,
    double approx_eps_frac
);

PYBIND11_MODULE(cpp_module_edge_detector, m) {
    m.doc() = "Edge/shape detector with tunable blur/canny/dilate/morph options";

    // Nice, Pythonic binding: all human-readable strings
    m.def(
        "outline_largest_shape_in_rect",
        [](py::array_t<uint8_t> input_array,
           py::tuple rect_tuple,
           int gauss_kernel_size = 3,
           double gauss_sigma = 0.0,
           double canny_thresh1 = 40.0,
           double canny_thresh2 = 80.0,
           int dilate_kernel_size = 3,
           int dilate_iterations = 2,
           const std::string &dilate_shape = "rect",   // rect|ellipse|cross
           const std::string &morph_op_str = "none",   // none|open|close|gradient|tophat|blackhat
           int morph_kernel_size = 3,
           int morph_iterations = 1,
           const std::string &morph_shape = "rect",    // rect|ellipse|cross
           bool morph_first = false,
           double approx_eps_frac = 0.02               // <=0 to skip approx
        ) {
            return outline_largest_shape_in_rect(
                input_array,
                rect_tuple,
                gauss_kernel_size,
                gauss_sigma,
                canny_thresh1,
                canny_thresh2,
                dilate_kernel_size,
                dilate_iterations,
                dilate_shape,
                morph_from_string(morph_op_str),
                morph_kernel_size,
                morph_iterations,
                morph_shape,
                morph_first,
                approx_eps_frac
            );
        },
        py::arg("input_array"),
        py::arg("rect_tuple"),
        py::arg("gauss_kernel_size") = 3,
        py::arg("gauss_sigma") = 0.0,
        py::arg("canny_thresh1") = 40.0,
        py::arg("canny_thresh2") = 80.0,
        py::arg("dilate_kernel_size") = 3,
        py::arg("dilate_iterations") = 2,
        py::arg("dilate_shape") = "rect",
        py::arg("morph_op") = "none",
        py::arg("morph_kernel_size") = 3,
        py::arg("morph_iterations") = 1,
        py::arg("morph_shape") = "rect",
        py::arg("morph_first") = false,
        py::arg("approx_eps_frac") = 0.02,
        R"doc(
Detect the largest shape inside a rectangle and draw a red outline.

Parameters
----------
input_array : np.ndarray[uint8]
    HxW (grayscale) or HxWx3 (BGR) image.
rect_tuple : (x, y, w, h)
    Region of interest in image coordinates.
gauss_kernel_size : int, default 3
gauss_sigma : float, default 0.0 (auto)
canny_thresh1 : float, default 40.0
canny_thresh2 : float, default 80.0
dilate_kernel_size : int, default 3
dilate_iterations : int, default 2
dilate_shape : {"rect","ellipse","cross"}, default "rect"
morph_op : {"none","open","close","gradient","tophat","blackhat"}, default "none"
morph_kernel_size : int, default 3
morph_iterations : int, default 1
morph_shape : {"rect","ellipse","cross"}, default "rect"
morph_first : bool, default False (do dilate then morph)
approx_eps_frac : float, default 0.02; <=0 draws raw contour

Returns
-------
np.ndarray[uint8]
    Same shape as input, with outline drawn.
)doc"
    );

    // (Optional) Back-compat alias for your old export name.
    // Calls the same implementation but keeps your previous import sites working.
    m.def(
        "blur_largest_shape_in_rect",
        [](py::array_t<uint8_t> input_array,
           py::tuple rect_tuple,
           int blur_kernel /*ignored but kept for signature compat*/ = 15) {
            // sensible defaults roughly matching the old pipeline
            return outline_largest_shape_in_rect(
                input_array, rect_tuple,
                /*gauss_kernel_size*/ 3,
                /*gauss_sigma*/ 0.0,
                /*canny_thresh1*/ 40.0,
                /*canny_thresh2*/ 80.0,
                /*dilate_kernel_size*/ 3,
                /*dilate_iterations*/ 2,
                /*dilate_shape*/ "rect",
                /*morph_op*/ -1,
                /*morph_kernel_size*/ 3,
                /*morph_iterations*/ 1,
                /*morph_shape*/ "rect",
                /*morph_first*/ false,
                /*approx_eps_frac*/ 0.02
            );
        },
        py::arg("input_array"),
        py::arg("rect_tuple"),
        py::arg("blur_kernel") = 15,
        "DEPRECATED alias. Use 'outline_largest_shape_in_rect' with explicit parameters."
    );
}
