import onnx
import torch
import numpy as np

MODEL_PATH = "model/pytorch/test1.pt"
DEVICE = 'cpu'
ONNX_MODEL_OUT = "model/onnx/test1.onnx"



if __name__ == "__main__":
    # Load model
    model = torch.load(MODEL_PATH)
    model.eval()

    # Test inference
    x_test = torch.tensor([[1]], dtype=torch.float,device=DEVICE)
    with torch.no_grad():
        model(x_test)
    
    # Convert to onnx
    input_name = ['input']
    output_name = ['output']
    input_set = torch.randn((1,1), device=DEVICE)
    torch.onnx.export(model, 
                      input_set, 
                      ONNX_MODEL_OUT,
                      input_names=input_name,
                      output_names=output_name)
    
    # Try to load
    onnx_model = onnx.load(ONNX_MODEL_OUT)
    onnx.checker.check_model(onnx_model)