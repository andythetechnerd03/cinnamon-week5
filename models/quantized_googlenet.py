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
    return Quantized_Googlenet(model)