import torch
from torch import nn
from models.googlenet import GoogleNet

class Quantized_Googlenet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantize the input tensor then feed into the model then de-quantize it.
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)

        return x
    
def quantized_googlenet(model):
    quantized_model = Quantized_Googlenet(model)

    quantized_model.eval()

    quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    
    modules_to_fuse = [
    ['model.prelayer.0', 'model.prelayer.1', 'model.prelayer.2'],
    ['model.prelayer.3', 'model.prelayer.4', 'model.prelayer.5'],
    ['model.prelayer.6', 'model.prelayer.7', 'model.prelayer.8'],

    ['model.a3.b1.0', 'model.a3.b1.1', 'model.a3.b1.2'],
    ['model.a3.b2.0', 'model.a3.b2.1', 'model.a3.b2.2'],
    ['model.a3.b2.3', 'model.a3.b2.4', 'model.a3.b2.5'],
    ['model.a3.b3.0', 'model.a3.b3.1', 'model.a3.b3.2'],
    ['model.a3.b3.3', 'model.a3.b3.4', 'model.a3.b3.5'],
    ['model.a3.b3.6', 'model.a3.b3.7', 'model.a3.b3.8'],
    ['model.a3.b4.1', 'model.a3.b4.2', 'model.a3.b4.3'],

    ['model.b3.b1.0', 'model.b3.b1.1', 'model.b3.b1.2'],
    ['model.b3.b2.0', 'model.b3.b2.1', 'model.b3.b2.2'],
    ['model.b3.b2.3', 'model.b3.b2.4', 'model.b3.b2.5'],
    ['model.b3.b3.0', 'model.b3.b3.1', 'model.b3.b3.2'],
    ['model.b3.b3.3', 'model.b3.b3.4', 'model.b3.b3.5'],
    ['model.b3.b3.6', 'model.b3.b3.7', 'model.b3.b3.8'],
    ['model.b3.b4.1', 'model.b3.b4.2', 'model.b3.b4.3'],

    ['model.a4.b1.0', 'model.a4.b1.1', 'model.a4.b1.2'],
    ['model.a4.b2.0', 'model.a4.b2.1', 'model.a4.b2.2'],
    ['model.a4.b2.3', 'model.a4.b2.4', 'model.a4.b2.5'],
    ['model.a4.b3.0', 'model.a4.b3.1', 'model.a4.b3.2'],
    ['model.a4.b3.3', 'model.a4.b3.4', 'model.a4.b3.5'],
    ['model.a4.b3.6', 'model.a4.b3.7', 'model.a4.b3.8'],
    ['model.a4.b4.1', 'model.a4.b4.2', 'model.a4.b4.3'],

    ['model.b4.b1.0', 'model.b4.b1.1', 'model.b4.b1.2'],
    ['model.b4.b2.0', 'model.b4.b2.1', 'model.b4.b2.2'],
    ['model.b4.b2.3', 'model.b4.b2.4', 'model.b4.b2.5'],
    ['model.b4.b3.0', 'model.b4.b3.1', 'model.b4.b3.2'],
    ['model.b4.b3.3', 'model.b4.b3.4', 'model.b4.b3.5'],
    ['model.b4.b3.6', 'model.b4.b3.7', 'model.b4.b3.8'],
    ['model.b4.b4.1', 'model.b4.b4.2', 'model.b4.b4.3'],

    ['model.c4.b1.0', 'model.c4.b1.1', 'model.c4.b1.2'],
    ['model.c4.b2.0', 'model.c4.b2.1', 'model.c4.b2.2'],
    ['model.c4.b2.3', 'model.c4.b2.4', 'model.c4.b2.5'],
    ['model.c4.b3.0', 'model.c4.b3.1', 'model.c4.b3.2'],
    ['model.c4.b3.3', 'model.c4.b3.4', 'model.c4.b3.5'],
    ['model.c4.b3.6', 'model.c4.b3.7', 'model.c4.b3.8'],
    ['model.c4.b4.1', 'model.c4.b4.2', 'model.c4.b4.3'],

    ['model.d4.b1.0', 'model.d4.b1.1', 'model.d4.b1.2'],
    ['model.d4.b2.0', 'model.d4.b2.1', 'model.d4.b2.2'],
    ['model.d4.b2.3', 'model.d4.b2.4', 'model.d4.b2.5'],
    ['model.d4.b3.0', 'model.d4.b3.1', 'model.d4.b3.2'],
    ['model.d4.b3.3', 'model.d4.b3.4', 'model.d4.b3.5'],
    ['model.d4.b3.6', 'model.d4.b3.7', 'model.d4.b3.8'],
    ['model.d4.b4.1', 'model.d4.b4.2', 'model.d4.b4.3'],

    ['model.e4.b1.0', 'model.e4.b1.1', 'model.e4.b1.2'],
    ['model.e4.b2.0', 'model.e4.b2.1', 'model.e4.b2.2'],
    ['model.e4.b2.3', 'model.e4.b2.4', 'model.e4.b2.5'],
    ['model.e4.b3.0', 'model.e4.b3.1', 'model.e4.b3.2'],
    ['model.e4.b3.3', 'model.e4.b3.4', 'model.e4.b3.5'],
    ['model.e4.b3.6', 'model.e4.b3.7', 'model.e4.b3.8'],
    ['model.e4.b4.1', 'model.e4.b4.2', 'model.e4.b4.3'],

    ['model.a5.b1.0', 'model.a5.b1.1', 'model.a5.b1.2'],
    ['model.a5.b2.0', 'model.a5.b2.1', 'model.a5.b2.2'],
    ['model.a5.b2.3', 'model.a5.b2.4', 'model.a5.b2.5'],
    ['model.a5.b3.0', 'model.a5.b3.1', 'model.a5.b3.2'],
    ['model.a5.b3.3', 'model.a5.b3.4', 'model.a5.b3.5'],
    ['model.a5.b3.6', 'model.a5.b3.7', 'model.a5.b3.8'],
    ['model.a5.b4.1', 'model.a5.b4.2', 'model.a5.b4.3'],

    ['model.b5.b1.0', 'model.b5.b1.1', 'model.b5.b1.2'],
    ['model.b5.b2.0', 'model.b5.b2.1', 'model.b5.b2.2'],
    ['model.b5.b2.3', 'model.b5.b2.4', 'model.b5.b2.5'],
    ['model.b5.b3.0', 'model.b5.b3.1', 'model.b5.b3.2'],
    ['model.b5.b3.3', 'model.b5.b3.4', 'model.b5.b3.5'],
    ['model.b5.b3.6', 'model.b5.b3.7', 'model.b5.b3.8'],
    ['model.b5.b4.1', 'model.b5.b4.2', 'model.b5.b4.3']
    ]

    quantized_model_fused = torch.ao.quantization.fuse_modules(quantized_model, modules_to_fuse)

    quantized_model = torch.ao.quantization.prepare(quantized_model_fused)

    model_int8 = torch.ao.quantization.convert(quantized_model)

    return model_int8, quantized_model