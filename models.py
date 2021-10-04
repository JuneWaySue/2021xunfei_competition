from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Linear

try:
    import timm
except:
    ! pip install timm
    import timm


class TimmModels(Module):
    def __init__(self, model_name='resnext50_32x4d',pretrained=True, num_classes=137):
        super(TimmModels, self).__init__()
        self.m = timm.create_model(model_name,pretrained=pretrained)
        model_list = list(self.m.children())
        model_list[-1] = Linear(
            in_features=model_list[-1].in_features, 
            out_features=num_classes, 
            bias=True
        )
        self.m = Sequential(*model_list)
        
    def forward(self, image):
        out = self.m(image)
        return out