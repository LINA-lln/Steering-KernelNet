# full assembly of the sub-parts to form the complete net

from torch.autograd import Variable
from .unet_parts import *
import math

class KernelNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(KernelNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 512)
        self.up3 = up(576, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x3 = self.down1(x1)
        x = self.up3(x3, x1)
        x = self.outc(x)

        # constraint parms
        x1 = 6 * F.sigmoid(x[:, 0, :, :])
        x2 = 2 * math.pi * F.sigmoid(x[:,1,:,:]) - math.pi
        x3 = 6 * F.sigmoid(x[:, 2, :, :])
        # x3 = torch.add(F.relu(x[:, 2, :, :]), 0.01)
        x1 = x1.expand(1,-1,-1,-1)
        x2 = x2.expand(1, -1, -1, -1)
        x3 = x3.expand(1, -1, -1, -1)
        x1 = x1.transpose(0,1)
        x2 = x2.transpose(0, 1)
        x3 = x3.transpose(0, 1)

        x = torch.cat((x1,x2,x3), dim=1)

        return x


if __name__ == '__main__':
    net = KernelNet(n_channels=3, n_classes=3).cuda()  #
    print(net)
    input = Variable(torch.rand(8, 3, 256, 320)).cuda()  # .cuda()
    output = net(input)
    print(output.size())
