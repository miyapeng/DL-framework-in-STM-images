class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_CA, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2
                 use_ca: bool,         # True
                 drop_rate: float,
                 index: str,           # 1a, 2a, 2b, ...
                 width_coefficient: float):
        # self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        # self.expanded_c = self.input_c * expanded_ratio
        # self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_ca = use_ca
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index


from functools import partial

# invertedresidualconfig = partial(InvertedResidualConfig, width_coefficient=0.2)
# print(invertedresidualconfig(*[3, 32, 16, 1, 1, True, 0.1, 1]).kernel)

from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')
# model = EfficientNet.from_pretrained('efficientnet-b1')
# model = EfficientNet.from_pretrained('efficientnet-b2')
# model = EfficientNet.from_pretrained('efficientnet-b3')
# model = EfficientNet.from_pretrained('efficientnet-b4')
# model = EfficientNet.from_pretrained('efficientnet-b5')
# model = EfficientNet.from_pretrained('efficientnet-b6')
model = EfficientNet.from_pretrained('efficientnet-b7')
