# Image Spinner

Ever wondered how an image would look like, if you'd spun it on a disc?

Wonder no longer, just grab this wonderful simulation.

Works very well to view zoetropes!

## Usage

Install via pip: `pip3 install .` inside this folder.

or very cool:

`pip3 install git+https://github.com/Wellidontcare/spinner`

Run via commandline: `spinner [--torch]`

## Requirements

- Python ≥ 3.10
- A working OpenGL context (for the imgui-bundle GUI)

### GPU acceleration (optional)

The `--torch` flag uses PyTorch with `affine_grid`/`grid_sample` for accelerated batched GPU compositing.
PyPI's default `torch` wheel is CPU-only. To get CUDA support, install

torch **before** installing this package: https://pytorch.org/get-started/locally/
