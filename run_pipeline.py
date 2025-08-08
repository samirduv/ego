import argparse
import os
import cv2
import cpp_module_edge_detector

from typing import Tuple

def run_pipeline(
    input_folder: str,
    output_folder: str,
    rect_tuple: Tuple[int, int, int, int],
    *,
    # Gaussian
    gauss_kernel_size: int = 3,
    gauss_sigma: float = 0.0,
    # Canny
    canny_thresh1: float = 40.0,
    canny_thresh2: float = 80.0,
    # Dilation
    dilate_kernel_size: int = 3,
    dilate_iterations: int = 2,
    dilate_shape: str = "rect",            # "rect" | "ellipse" | "cross"
    # MorphologyEx
    morph_op: str = "none",                # "none" | "open" | "close" | "gradient" | "tophat" | "blackhat"
    morph_kernel_size: int = 3,
    morph_iterations: int = 1,
    morph_shape: str = "rect",             # "rect" | "ellipse" | "cross"
    morph_first: bool = False,
    # Drawing
    approx_eps_frac: float = 0.02,
    # Files
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
) -> None:
    """
    Process all images in input_folder, outlining the largest shape inside rect_tuple,
    and save to output_folder using the configured edge/morph settings.
    """

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(exts):
            continue

        in_path = os.path.join(input_folder, filename)
        print(f"Processing file: {filename}")

        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Could not read image: {filename}. Skipping.")
            continue

        height, width = img.shape[:2]
        x, y, w, h = rect_tuple

        # Clamp ROI to image bounds (and report if changed)
        if x < 0 or y < 0 or x + w > width or y + h > height:
            print(f"  ROI {rect_tuple} out of bounds for {filename}. Adjusting.")
            x = max(0, x)
            y = max(0, y)
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))
        copy_rect_tuple = (x, y, w, h)

        try:
            outlined = cpp_module_edge_detector.outline_largest_shape_in_rect(
                img,
                copy_rect_tuple,
                gauss_kernel_size,
                gauss_sigma,
                canny_thresh1,
                canny_thresh2,
                dilate_kernel_size,
                dilate_iterations,
                dilate_shape,       # "rect" | "ellipse" | "cross"
                morph_op,           # "none" | "open" | ...
                morph_kernel_size,
                morph_iterations,
                morph_shape,        # "rect" | "ellipse" | "cross"
                morph_first,
                approx_eps_frac
            )
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
            continue

        out_path = os.path.join(output_folder, filename)
        ok = cv2.imwrite(out_path, outlined)
        if ok:
            print(f"  Saved processed image to: {out_path}")
        else:
            print(f"  Failed to save: {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing pipeline (Python + C++)")

    # I/O
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output images folder")
    parser.add_argument("--rect", nargs=4, type=int, metavar=("X","Y","W","H"),
                        required=True, help="ROI rectangle: x y w h")

    # Gaussian
    parser.add_argument("--gauss-kernel", type=int, default=3,
                        help="Gaussian kernel size (odd int, e.g. 3,5,7)")
    parser.add_argument("--gauss-sigma", type=float, default=0.0,
                        help="Gaussian sigma (0 = auto from kernel size)")

    # Canny
    parser.add_argument("--canny-th1", type=float, default=40.0,
                        help="Canny lower threshold")
    parser.add_argument("--canny-th2", type=float, default=80.0,
                        help="Canny upper threshold")

    # Dilation
    parser.add_argument("--dilate-kernel", type=int, default=3,
                        help="Dilation kernel size")
    parser.add_argument("--dilate-iters", type=int, default=2,
                        help="Dilation iterations")
    parser.add_argument("--dilate-shape", choices=["rect","ellipse","cross"], default="rect",
                        help="Dilation kernel shape")

    # MorphologyEx
    parser.add_argument("--morph-op", choices=["none","open","close","gradient","tophat","blackhat"],
                        default="none", help="morphologyEx operation")
    parser.add_argument("--morph-kernel", type=int, default=3,
                        help="Morphology kernel size")
    parser.add_argument("--morph-iters", type=int, default=1,
                        help="Morphology iterations")
    parser.add_argument("--morph-shape", choices=["rect","ellipse","cross"], default="rect",
                        help="Morphology kernel shape")
    parser.add_argument("--morph-first", action="store_true",
                        help="Apply morphologyEx before dilation (default: after)")

    # Drawing
    parser.add_argument("--approx-eps-frac", type=float, default=0.02,
                        help="ApproxPolyDP epsilon as fraction of perimeter (<=0 to draw raw contour)")

    args = parser.parse_args()

    run_pipeline(
        input_folder=args.input,
        output_folder=args.output,
        rect_tuple=tuple(args.rect),
        gauss_kernel_size=args.gauss_kernel,
        gauss_sigma=args.gauss_sigma,
        canny_thresh1=args.canny_th1,
        canny_thresh2=args.canny_th2,
        dilate_kernel_size=args.dilate_kernel,
        dilate_iterations=args.dilate_iters,
        dilate_shape=args.dilate_shape,
        morph_op=args.morph_op,
        morph_kernel_size=args.morph_kernel,
        morph_iterations=args.morph_iters,
        morph_shape=args.morph_shape,
        morph_first=args.morph_first,
        approx_eps_frac=args.approx_eps_frac,
    )
