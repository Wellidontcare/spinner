import cv2
import numpy as np
from imgui_bundle import hello_imgui, imgui, immvision
from imgui_bundle import portable_file_dialogs as pfd
from pathlib import Path

class SpinnerApp:
    def __init__(self):
        self.image_params = immvision.ImageParams()
        self.runner_params = hello_imgui.RunnerParams()
        self.runner_params.app_window_params.window_title = "Image Spinner"
        self.runner_params.app_window_params.window_geometry.size = (720, 620)
        self.runner_params.callbacks.show_gui = self.frame
        self.runner_params.fps_idling.enable_idling = False
        self.image_params = immvision.ImageParams(refresh_image=True)
        self.spin_image = np.zeros((512, 512, 3), np.uint8)
        self.original = self.spin_image.copy()
        self.big_original = self.spin_image.copy()
        self.image_file = None
        self.frame_counter = 0
        self.speed = 1
        self.clockwise = False
        immvision.use_bgr_color_order()


    def frame(self):
        available_width = imgui.get_content_region_avail().x

        imgui.spacing()
        imgui.separator_text("Image")

        h, w = self.spin_image.shape[:2]
        direction = -1 if self.clockwise else 1
        rot_mat = cv2.getRotationMatrix2D((w/2, h/2), direction*self.frame_counter*self.speed, 1)
        self.spin_image = cv2.warpAffine(self.original, rot_mat, (w, h))



        button_width = available_width * 0.5 - imgui.get_style().item_spacing.x * 0.5
        if imgui.button("Load Image", imgui.ImVec2(button_width, 0)):
            self.image_file = pfd.open_file("Open Image File", Path().home().as_posix())
            self.spin_image = cv2.imread(self.image_file.result().pop(), cv2.IMREAD_COLOR)
            self.big_original = self.spin_image.copy()
            h, w = self.spin_image.shape[:2]
            ratio = w / h
            self.spin_image = cv2.resize(self.spin_image, (int(200*ratio), 200))
            self.original = self.spin_image.copy()

        imgui.same_line()
        if imgui.button("Create Image", imgui.ImVec2(button_width, 0)):
            self.image = self.original.copy()
            image_count = int(360 // self.speed) + 1
            rot_mat = cv2.getRotationMatrix2D((w/2, h/2), 360/image_count, 1)
            h, w, _ = self.big_original.shape
            self.image = self.big_original.copy().astype(np.float64)
            self.image2 = cv2.warpAffine(self.image, rot_mat, (w, h))

            direction = -1 if self.clockwise else 1
            for i in range(1, 20*(image_count)):
                self.image += self.image2
                rot_mat = cv2.getRotationMatrix2D((w/2, h/2), direction*i*(360/image_count), 1)
                self.image2 = cv2.warpAffine(self.big_original, rot_mat, (w, h))
            self.image[:, :, 0] = 255*(self.image[:, :, 0]/self.image[:, :, 0].max())
            self.image[:, :, 1] = 255*(self.image[:, :, 1]/self.image[:, :, 1].max())
            self.image[:, :, 2] = 255*(self.image[:, :, 2]/self.image[:, :, 2].max())
            self.image = self.image.astype(np.uint8)
            save_location = pfd.save_file("Save image", Path().home().as_posix())
            cv2.imwrite(save_location.result(), self.image)
        
        _, self.clockwise = imgui.checkbox("Clockwise", self.clockwise)

        _, self.speed = imgui.slider_float("Speed", self.speed, 1.0, 180.0)
       
        self.image_params.image_display_size = (
            int(available_width), int(available_width * h / max(w, 1))
        )
        immvision.image("##spinimage", self.spin_image, self.image_params)

        self.frame_counter += 1
        self.frame_counter = self.frame_counter % 360


    def run(self):
        hello_imgui.run(self.runner_params)


def main():
    app = SpinnerApp()
    app.run()


if __name__ == "__main__":
    main()