import onnxruntime as ort
import multiprocessing


class PredictBase(object):
    def __init__(self):
        pass

    def get_onnx_session(self, model_dir, use_gpu, gpu_id=0):
        # 使用gpu
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

        onnx_session = ort.InferenceSession(model_dir, sess_options=so, providers=providers)

        # print("providers:", onnxruntime.get_device())
        return onnx_session

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_numpy
        return input_feed
