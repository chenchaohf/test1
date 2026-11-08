import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV2, self).__init__()
        # 加载ImageNet预训练MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # 默认全部参数可训练
        for param in self.model.parameters():
            param.requires_grad = True
        # 替换分类头
        in_channels = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = MobileNetV2(num_classes=5)
    out = model(x)
    print("输出shape:", out.shape)