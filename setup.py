from setuptools import setup, Extension
import pybind11
import numpy

ext_modules = [
    Extension(
        "cpp_module_edge_detector",
        ["cpp_module/cpp_module.cpp"],  # Corrected path
        include_dirs=[
            pybind11.get_include(),
            numpy.get_include(),
            "/usr/include/opencv4"
        ],
        language="c++",
        extra_compile_args=["-O3", "-std=c++11"],
        libraries=["opencv_core", "opencv_imgproc", "opencv_imgcodecs"],
    ),
]

setup(
    name="cpp_module",
    version="0.1",
    ext_modules=ext_modules,
)
