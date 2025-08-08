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

        # check if the rectangle is larger than the image dimensions
        if img is None:
            print(f"Could not read image: {filename}. Skipping.")
            continue

        height, width = img.shape[:2]
        x, y, w, h = rect_tuple

        if x < 0 or y < 0 or x + w > width or y + h > height:

            print(f"Rectangle {rect_tuple} is out of bounds for image {filename}. Reducing rectangle to fit.")
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            w = max(0, w)
            h = max(0, h)
            copy_rect_tuple = (x, y, w, h)

            print(f"Adjusted rectangle to: {copy_rect_tuple}")

        blurred_img = cpp_module_edge_detector.blur_largest_shape_in_rect(
            img,
            copy_rect_tuple,
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
