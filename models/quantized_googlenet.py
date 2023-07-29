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

    quantized_model = torch.ao.quantization.prepare(quantized_model)

    model_int8 = torch.ao.quantization.convert(quantized_model)

    return model_int8, quantized_model