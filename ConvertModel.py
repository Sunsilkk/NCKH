import torch
import timm

device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
model = timm.create_model('efficientnet_b0')
model = model.to(device)

input = torch.rand(1,3,224,224).to(device)
OutputPath = "./ONNXModels/Efficientnet.onnx"

torch.onnx.export(model, #Your model
                input, #Model input
                OutputPath, #Output path to save model
                opset_version=12, #ONNX version
                do_constant_folding=False, #We will use another library
                export_params=True, #Export parameters
                input_names = ['Input'], 
                output_names = ['Output'],)
