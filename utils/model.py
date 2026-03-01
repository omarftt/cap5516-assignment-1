import torch.nn as nn
import timm


class ViTTiny(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.vit = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=2,
        )

    def forward(self, x):
        return self.vit(x)

    def get_last_block(self):
        """Middle transformer block """
        return self.vit.blocks[len(self.vit.blocks) // 2]