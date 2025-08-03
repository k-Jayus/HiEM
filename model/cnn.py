import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm3d(out_ch),
            #nn.LeakyReLU(),
            #nn.Conv3d(out_ch, out_ch, 3, 1, 1),
            #nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(),
            nn.MaxPool3d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


class Net(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        ch = [1, 16, 32, 64, 128, 256, 512]
        self.block = nn.Sequential(*[Block(ch[idx - 1], ch[idx]) for idx in range(1, len(ch))])
        self.ln = nn.Linear(ch[-1] + 26, 2)
        self.ln2 = nn.Linear(64, num_classes)



    def forward(self, x, x2):
        out = self.block(x)
        out = out.sum(dim=(2, 3, 4))
        out = torch.cat((out, x2), dim=1)
        out = self.ln(out)
        #out = self.ln2(out)

        return out


if __name__ == "__main__":
    x = torch.rand(1, 1, 182, 218, 182).to(torch.float32)
    x2 = torch.rand(1, 135)
    net = Net()
    y = net(x, x2)
    print(y.shape)
