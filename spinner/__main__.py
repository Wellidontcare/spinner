import argparse
from spinner.spinner_live import SpinnerApp
try:
    import torch
    from spinner.spinner_torch import SpinnerAppTorch
except ImportError:
    print("You have no torch installed, torch install is optional and not configured via pyproject")
    SpinnerAppTorch = SpinnerApp


def main():
    argparser = argparse.ArgumentParser("spinner", description="An application that can spin images!")
    argparser.add_argument("--torch", action="store_true", help="Run the gpu accelerated version (which is not live)")

    args = argparser.parse_args()

    if args.torch:
        app = SpinnerAppTorch()
    else:
        app = SpinnerApp()

    app.run()