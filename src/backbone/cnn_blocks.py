from ..layers.conv_block import ConvBlock

def make_vgg_block(in_ch, out_ch, num_conv=2):
    layers = []
    for _ in range(num_conv):
        layers.append(ConvBlock(in_ch, out_ch))
        in_ch = out_ch
    return layers
