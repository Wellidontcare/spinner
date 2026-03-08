import cv2
import numpy as np
import torch
import torch.nn.functional as F
from imgui_bundle import hello_imgui, imgui, immvision
from imgui_bundle import portable_file_dialogs as pfd
from pathlib import Path
from numpy.typing import NDArray

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.squeeze(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).byte().cpu().numpy()

def _build_composite_gpu(src: torch.Tensor, n: int, chunk: int = 16) -> torch.Tensor:
    _, c, h, w = src.shape
    accumulator = torch.zeros(1, c, h, w, device=src.device, dtype=src.dtype)

    angles = torch.arange(n, device=DEVICE, dtype=torch.float32) * (360.0 / n)

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        batch_size = end - start
        batch_angles = angles[start:end]

        rads = torch.deg2rad(batch_angles)
        cos_a = torch.cos(rads)
        sin_a = torch.sin(rads)
        ar = w / h
        zeros = torch.zeros(batch_size, device=src.device)

        # B×2×3
        thetas = torch.stack([
            torch.stack([cos_a, -sin_a * ar, zeros], dim=1),
            torch.stack([sin_a / ar, cos_a, zeros], dim=1),
        ], dim=1)

        src_batch = src.expand(batch_size, -1, -1, -1)
        grids = F.affine_grid(thetas, src_batch.shape, align_corners=False)
        rotated = F.grid_sample(src_batch, grids, mode="bilinear",
                                padding_mode="zeros", align_corners=False)
        accumulator += rotated.sum(dim=0, keepdim=True)

    # normalise
    amax = accumulator.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    accumulator /= amax
    return accumulator


def _to_device(img_bgr: NDArray) -> torch.Tensor:
    """HWC uint8 numpy → 1CHW float32 tensor on device, range [0,1]."""
    t = torch.from_numpy(img_bgr).float().div(255.0)
    t = t.permute(2, 0, 1).unsqueeze(0)
    return t.to(DEVICE)


class SpinnerAppTorch:
    def __init__(self):
        print(torch.cuda.is_available())
        self.params = hello_imgui.RunnerParams()
        self.params.app_window_params.window_title = "Image Spinner"
        self.params.app_window_params.window_geometry.size = (720, 620)
        self.params.callbacks.show_gui = self.frame
        self.params.fps_idling.enable_idling = True

        self.image_params = immvision.ImageParams()
        self.image_params.refresh_image = True

        self.display_size = 512
        self.placeholder = np.zeros((self.display_size, self.display_size, 3), np.uint8)
        self.original_image = None
        self.gpu_image = None 
        self.composite_display = self.placeholder
        self.step_count = 6
        self.dirty = True
        immvision.use_bgr_color_order()

    def load_image(self):
        dialog = pfd.open_file("Open Image", Path.home().as_posix())
        paths = dialog.result()
        if not paths:
            return
        img = cv2.imread(paths.pop(), cv2.IMREAD_COLOR)
        if img is None:
            return
        self.original_image = img
        h, w, _ = img.shape
        self.gpu_image = _to_device(cv2.resize(img, (1000, int(1000*h/w))))
        self.dirty = True

    def rebuild_composite(self):
        if self.gpu_image is None:
            self.composite_display = self.placeholder
            return
        n = max(self.step_count, 1)
        composite = _build_composite_gpu(self.gpu_image, n)
        composite_ndarray = _to_numpy(composite)
        h, w = composite_ndarray.shape[:2]
        scale = self.display_size / max(h, w)
        self.composite_display = cv2.resize(
            composite_ndarray, (int(w * scale), int(h * scale))
        )

    def save_composite(self):
        if self.gpu_image is None:
            return
        n = max(self.step_count, 1)
        composite = _build_composite_gpu(_to_device(self.original_image), n)
        composite_ndarray = _to_numpy(composite)
        dialog = pfd.save_file("Save Composite", Path.home().as_posix())
        path = dialog.result()
        if path:
            cv2.imwrite(path, composite_ndarray)

    def frame(self):
        available_width = imgui.get_content_region_avail().x

        imgui.spacing()
        imgui.separator_text("Image")

        button_width = available_width * 0.5 - imgui.get_style().item_spacing.x * 0.5
        if imgui.button("Load Image", imgui.ImVec2(button_width, 0)):
            self.load_image()
        imgui.same_line()
        
        has_image = self.gpu_image is not None
        if not has_image:
            imgui.begin_disabled()
        if imgui.button("Save Composite", imgui.ImVec2(button_width, 0)):
            self.save_composite()
        if not has_image:
            imgui.end_disabled()

        imgui.spacing()
        imgui.separator_text("Parameters")

        imgui.set_next_item_width(-1)
        changed, self.step_count = imgui.slider_int(
            "##steps", self.step_count, 1, 72, "Rotation steps: %d"
        )
        if changed:
            self.dirty = True

        device_label = "CUDA" if DEVICE == "cuda" else "CPU (no CUDA)"
        imgui.text_disabled(f"Backend: {device_label}")

        if self.dirty:
            self.rebuild_composite()
            self.dirty = False

        imgui.spacing()
        imgui.separator_text("Preview")

        h, w = self.composite_display.shape[:2]
        self.image_params.image_display_size = (
            int(available_width), int(available_width * h / max(w, 1))
        )
        immvision.image(
            "##preview", self.composite_display, self.image_params
        )

    def run(self):
        hello_imgui.run(self.params)

if __name__ == "__main__":
    SpinnerAppTorch().run()
