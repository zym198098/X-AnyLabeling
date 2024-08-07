import logging
import os

import cv2
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication

from anylabeling.app_info import __preferred_device__
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from .model import Model
from .types import AutoLabelingResult
from .engines.build_onnx_engine import OnnxBaseModel
# from .utils.points_conversion import cxywh2xyxy
import debugpy

class Threshold_jzw(Model):
    """Object detection model using SegFormer"""

    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config["type"]
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        self.classes = self.config.get("classes", [])
       
        self.net = None
        self.classes = self.config["classes"]
        self.input_shape = None

    def preprocess(self, input_image):
        """
        Pre-processes the input image before feeding it to the network.

        Args:
            input_image (numpy.ndarray): The input image to be processed.

        Returns:
            numpy.ndarray: The pre-processed output.
        """
        debugpy.debug_this_thread()
        # Get the image width and height
        # 将彩色图像转换为灰度图像  
        image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  

        return image

    def postprocess(self, input_image, outputs):
        """
        Post-processes the network's output.

        Args:
            input_image (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output from the network.

        Returns:
            list: List of dictionaries containing the output
                    approx_contours
        """
        debugpy.debug_this_thread()
        # nonzero_indices1 = np.flatnonzero(outputs)
  
        # cv2.imwrite("post.png",img_tem)
        ret,masks=cv2.threshold(input_image,100,255,cv2.THRESH_TRIANGLE)
       
        # cv2.imwrite("post255.png",masks)
        masks=~masks
        # cv2.imshow("input",masks)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        contours, _ = cv2.findContours(
                masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area < image_size * 0.9
            ]

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour
                for contour, area in zip(approx_contours, areas)
                if area > 25#if area > avg_area * 0.02
            ]
            approx_contours = filtered_approx_contours
            # shapes = []
            # output_mode= "polygon"
            # if output_mode == "polygon":
            #     for approx in approx_contours:i
            #         # Scale points
            #         points = approx.reshape(-1, 2)
            #         points[:, 0] = points[:, 0]
            #         points[:, 1] = points[:, 1]
            #         points = points.tolist()
            #         if len(points) < 3:
            #             continue
            #         points.append(points[0])

            #         # Create shape
            #         shape = Shape(flags={})
            #         for point in points:
            #             point[0] = int(point[0])
            #             point[1] = int(point[1])
            #             shape.add_point(QtCore.QPointF(point[0], point[1]))
            #         shape.shape_type = "polygon"
            #         shape.closed = True
            #         shape.fill_color = "#000000"
            #         shape.line_color = "#000000"
            #         shape.line_width = 1
            #         shape.label = "AUTOLABEL_OBJECT"
            #         shape.selected = False
            #         shapes.append(shape)   
        return approx_contours

    def predict_shapes(self, image, image_path=None):
        """
        Predict shapes from image
        """
        debugpy.debug_this_thread()

        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, image_path)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []

        blob = self.preprocess(image)
        detections=None
     
        approx_contours = self.postprocess(blob, detections)
        shapes = []
        output_mode= "polygon"
        if self.output_mode == "polygon":
            for approx in approx_contours:
                # Scale points
                points = approx.reshape(-1, 2)
                points[:, 0] = points[:, 0]
                points[:, 1] = points[:, 1]
                points = points.tolist()
                if len(points) < 3:
                    continue
                points.append(points[0])

                # Create shape
                shape = Shape(flags={})
                for point in points:
                    point[0] = int(point[0])
                    point[1] = int(point[1])
                    shape.add_point(QtCore.QPointF(point[0], point[1]))
                shape.shape_type = "polygon"
                shape.closed = True
                shape.fill_color = "#000000"
                shape.line_color = "#000000"
                shape.line_width = 1
                shape.label = self.classes[0]
                shape.selected = False
                shapes.append(shape)  

        result = AutoLabelingResult(shapes, replace=True)
        return result

    def unload(self):

        debugpy.debug_this_thread()
        if self.net:
            del self.net
