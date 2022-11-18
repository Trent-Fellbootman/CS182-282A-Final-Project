from flax import linen

class ConvCustom(linen.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        super(ConvCustom, self).__init__()
        self.conv_layer = linen.Conv(features=out_channels, kernel_size=kernel_size, strides=stride, padding=padding)
        
        if batch_norm:
            self.batchnorm = linen.BatchNorm()
