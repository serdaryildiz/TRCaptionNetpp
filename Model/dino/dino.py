import numpy
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

preprocess = transforms.Compose([transforms.Resize((224, 224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])


class DinoV2(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.vision_encoder = torch.hub.load('facebookresearch/dinov2', model_name)
        self.vision_encoder = self.vision_encoder.eval().cuda().half()
        return

    def forward(self, x):
        return self.vision_encoder.forward_features(x)['x_norm_patchtokens']

    def get_output_dim(self):
        with torch.no_grad():
            dummpy_input_image = preprocess(Image.fromarray(numpy.zeros((512, 512, 3), dtype=numpy.uint8))).to(
                next(self.parameters()).device).half()
            encoder_output_size = self.vision_encoder(dummpy_input_image.unsqueeze(0)).shape[-1]
        return encoder_output_size