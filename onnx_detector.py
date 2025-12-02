import cv2
import numpy as np
import onnxruntime as ort
from logger_util import MyLogger
import multiprocessing

model_logger = MyLogger().get_tag_logger("model")


def convert_xywh_to_xyxy(box):
    x1, y1, w, h = box
    return [x1, y1, x1 + w, y1 + h]


class ONNX_Detector:
    def __init__(self, onnx_model_path: str, input_shape=640, confidence_thres: float = 0.25):
        so = ort.SessionOptions()
        num_threads = max(1, multiprocessing.cpu_count() // 2)
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = num_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            }),
            'CPUExecutionProvider'
        ]

        self.session = ort.InferenceSession(onnx_model_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_width = self.input_height = input_shape
        self.confidence_thres = confidence_thres

        self._warm_up_model()
        self._check_inference_device()

    def _warm_up_model(self):
        try:
            dummy = np.zeros((1, 3, self.input_height, self.input_width), dtype=np.float32)
            for _ in range(3):
                _ = self.session.run(None, {self.input_name: dummy})
            model_logger.info("[ONNX_Detector] 模型预热完成")
        except Exception as e:
            model_logger.warning(f"模型预热失败: {e}")

    def _check_inference_device(self):
        providers = self.session.get_providers()
        device = "GPU" if 'CUDAExecutionProvider' in providers else "CPU"
        model_logger.info(f"[ONNX_Detector] 模型运行设备: {device}")

    def predict(self, img: np.ndarray) -> list:

        target_size = (640, 640)
        h, w = img.shape[:2]
        scale = min(target_size[0] / w, target_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        pad_w = target_size[0] - new_w
        pad_h = target_size[1] - new_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        h, w = img_padded.shape[:2]
        # 3. 归一化
        img_resized = img_padded.astype(np.float32) / 255.0
        img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC -> CHW
        img_resized = np.expand_dims(img_resized, axis=0)  # NCHW # 4.
        # 构造 im_shape 和 scale_factor
        im_shape = np.array([[h, w]], dtype=np.float32)  # 原始图像大小
        scale_factor = np.array([[target_size[1] / w, target_size[0] / h]], dtype=np.float32)  # 宽/高缩放比例
        input_name = []
        for node in self.session.get_inputs():
            input_name.append(node.name)

        output_name = []
        for node in self.session.get_outputs():
            output_name.append(node.name)
        inputs = {
            "image": img_resized,
            "im_shape": im_shape,
            "scale_factor": scale_factor
        }
        outputs = self.session.run(None, inputs)
        preds = outputs[0]  # shape: (num_boxes, 4 + num_classes)
        conf_mask = preds[:, 1] >= self.confidence_thres
        preds = preds[conf_mask]
        if preds.shape[0] == 0:
            return []
        boxes = preds[:, 2:]
        boxes[:, 0] = (boxes[:, 0] - left) / scale  # x1
        boxes[:, 2] = (boxes[:, 2] - left) / scale  # x2
        boxes[:, 1] = (boxes[:, 1] - top) / scale  # y1
        boxes[:, 3] = (boxes[:, 3] - top) / scale  # y2

        stats = {
            "max_w": -1e9,
            "max_h": -1e9,
            "min_w": 1e9,
            "min_h": 1e9,
            "max_area": -1e9,
            "min_area": 1e9,
        }

        filtered_boxes = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            w = x2 - x1
            h = y2 - y1
            area = w * h

            # ----------- 区间过滤 -----------
            if not (20 <= w <= 200 and 20 <= h <= 100):
                continue
            # ----------------------------------

            filtered_boxes.append([x1, y1, x2, y2])

            # 更新统计值
            stats["max_w"] = max(stats["max_w"], w)
            stats["max_h"] = max(stats["max_h"], h)
            stats["min_w"] = min(stats["min_w"], w)
            stats["min_h"] = min(stats["min_h"], h)
            stats["max_area"] = max(stats["max_area"], area)
            stats["min_area"] = min(stats["min_area"], area)

            # 绘制框
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        print("==== 统计结果 ====")
        print("最大宽度:", stats["max_w"])
        print("最大高度:", stats["max_h"])
        print("最小宽度:", stats["min_w"])
        print("最小高度:", stats["min_h"])
        print("最大面积:", stats["max_area"])
        print("最小面积:", stats["min_area"])

        return filtered_boxes



# doc_layout_model = ONNX_Detector("G:\don\ocr_number_demo\RT-DETR-L_wired_table_cell_det\inference.onnx",
#                                       640,
#                                       0.9)
#
#
#
# doc_layout_model.predict(cv2.imread("1.jpg"))
