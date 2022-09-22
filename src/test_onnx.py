import onnxruntime as ort
import numpy as np

ONNX_MODEL_PATH = 'model/onnx/test1.onnx'

x = np.asarray([[1]])
ort_sess = ort.InferenceSession(ONNX_MODEL_PATH)
outputs = ort_sess.run(None, {"input" : x.astype(np.float32)})
print(outputs[0][0][0])