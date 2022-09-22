import argparse
import numpy as np
import onnxruntime as ort

arg_parser = argparse.ArgumentParser(
    prog="2x+5",
    description="A 2x+5 programs that calculate using onnx model exported from pytorch model"
)

arg_parser.add_argument("--onnx-path", 
                        type=str, 
                        required=True,
                        help="A path to onnx model")

arg_parser.add_argument("-x", 
                        type=float, 
                        required=True,
                        help="A number that you want to put to the x")

args = arg_parser.parse_args()

onnx_model_path = args.onnx_path
ort_sess = ort.InferenceSession(onnx_model_path)
x = args.x
x = np.asarray([[x]]).astype(np.float32)
outputs = ort_sess.run(None, {"input" : x})
print(outputs[0][0][0])