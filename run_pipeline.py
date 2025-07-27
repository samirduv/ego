import argparse
import os
import cv2
import cpp_module

def run_pipeline(input_folder: str, output_folder: str, rect_tuple: tuple[int, int, int, int], blur_kernel: int = 15) -> None:
    # TODO: Implement this function
    # [candidate to fill]
    raise NotImplementedError("Not implemented")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing pipeline (Python + C++)")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output images folder")
    parser.add_argument("--rect", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), required=True, help="Rectangle (x y w h) for ROI")
    parser.add_argument("--blur_kernel", type=int, default=15, help="Blur kernel size (odd integer)")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, tuple(args.rect), args.blur_kernel)
