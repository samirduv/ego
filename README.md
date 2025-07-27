# Egonym SWE Take Home Technical Assignment

Welcome! This project is designed as a take-home technical assignment to evaluate your skills in:

1. **C++ / Python Integration**
2. **Some Image Processing**
3. **Little DevOps**

Please read the instructions carefully and follow the steps below to complete the assignment.

---

## Project Overview

You are provided with a scaffold of a hybrid C++/Python image processing pipeline. The core image processing logic is to be implemented in C++ (using OpenCV and pybind11), and exposed to Python. The pipeline detects the largest shape within a specified rectangle in each image and applies a blur to it.

**Development will be done through a Docker container.** By default, build and run the container locally on your own machine. Optionally, you can do a cloud development / deployment but you'll have to set up any cloud infrastructure yourself. If you do, please stick to the provided Dockerfile to ensure consistency.

Your tasks will cover:
- Implementing the image processing logic in C++ (using OpenCV and pybind11)
- Completing or modifying the Python pipeline script as described below

---

## Directory Structure

```
ego-swe-image-pipeline/
  ├── cpp_module/                # C++ source for pybind11 module
  │    └── [TODO]cpp_module.cpp
  ├── input_images/              # Place your input images here
  ├── output_images/             # Output images will be written here
  ├── [TODO]run_pipeline.py      # Main Python entrypoint
  ├── requirements.txt           # Python dependencies
  ├── setup.py                   # Build script for C++ extension
  ├── Dockerfile                 # Container setup
  └── README.md                  # This file
```

---

## Assignment Instructions

### 0. Setup

Create your own fork of this Egonym's public repository. In case you don't have a GitHub account, please create one. Besides this `README`, the repository contains some boilerplate and scaffolding to save you some headache and to make sure you spend time on what matters. The files that you'll need to modify are preppended with [TODO] in their respective filenames. The assignment consists of subtasks defined in the following sections.

### 1. Build and Run Development Container

**Build the image:**
```bash
docker build -t image-pipeline .
```

**Run the container:**
```bash
docker run --rm -it -v $(pwd):/home/devuser/workspace image-pipeline
```

---

### 2. Image Pipeline Solution

The problem you're supposed to solve is to blur the biggest detected shape in a user-defined rectangle. You'll implement two things: 1) Python-based pipeline runner (`[TODO]run_pipeline.py`) and 2) C++ module for the actual image processing (`cpp_module/[TODO]cpp_module.cpp`).

To be able to use the C++ module from the Python pipeline, you'll have to build it as an extension

```bash
python setup.py build_ext --inplace 
```

We've defined a simple Python CLI interface to use the pipeline

```bash
python run_pipeline.py \
  --input input_images \
  --output output_images \
  --rect 10 10 100 100 \
  --blur_kernel 15
```

- `--rect` specifies the rectangle (x, y, width, height) for the region of interest in each image.
- `--blur_kernel` sets the blur kernel size (must be an odd integer).

By default, source images will be taken from `input_images/` and output images will be saved in `output_images/`.

Additionally, we've defined function signatures to help you with the expected interfaces (both in C++ and Python).

---

## What We’re Looking For

- **Correctness:** The pipeline should build and run as described.
- **Code Quality:** Clean, readable, and well-documented code.
- **Communication:** Well-written documentation and commit messages.

Note: There are **no** extra points for **not** using AI coding tools. Instead, we encourage a very skeptical use of it subject to a great amount of scrutiny. We ask you to be transparent with us where and how much AI you've used while working on the assignment.

---

## Submission

- Please provide your code, documentation, and any notes in a public or private repository.
- The reviewer will run the pipeline on additional samples in a Docker container, either locally or in a cloud environment as appropriate.

---

## Good Luck!

If you have any questions or difficulties, please reach out.