import argparse
import os
import cv2
import cpp_module_edge_detector

def run_pipeline(input_folder: str, output_folder: str, rect_tuple: tuple[int, int, int, int], blur_kernel: int = 15) -> None:
    # TODO: Implement this function
    # [candidate to fill]
    for filename in os.listdir(input_folder):

        print(f"Processing file: {filename}")

        img = cv2.imread(os.path.join(input_folder, filename))

        blurred_img = cpp_module_edge_detector.blur_largest_shape_in_rect(
            img,
            rect_tuple,
            blur_kernel
        )

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, blurred_img)

        print(f"Saved processed image to: {output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing pipeline (Python + C++)")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output images folder")
    parser.add_argument("--rect", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), required=True, help="Rectangle (x y w h) for ROI")
    parser.add_argument("--blur_kernel", type=int, default=15, help="Blur kernel size (odd integer)")
    args = parser.parse_args()
    run_pipeline(args.input, args.output, tuple(args.rect), args.blur_kernel)
