from torch import block_diag, nn
from torch.nn import functional as F
from torchinfo import summary
import math
import pytorch_lightning as pl
import torch
import torchvision


class SENet_block(pl.LightningModule):
    expansion=1
    def __init__(self,inplanes, planes,stride=1,downsample=None) -> None:
        super().__init__()
        self.conv_part1=nn.Sequential(
                nn.Conv2d(inplanes,planes,kernel_size=3,stride=stride,padding=1),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(planes),
                )
        self.downsample=downsample
        self.stride=stride

        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(in_features=planes,out_features=planes//4)
        self.fc2=nn.Linear(in_features=planes//4,out_features=planes)

    def forward(self,x):
        residual = x
        out=self.conv_part1(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        #SE Block
        original_out=out
        out=self.globalAvgPool(out)
        out=out.view(out.size(0),-1)
        out=self.fc1(out)
        out=nn.ReLU()(out)
        out=self.fc2(out)
        out=nn.Sigmoid()(out)
        out=out.view(out.size(0),out.size(1),1,1)
        out=out*original_out
        out+=residual
        out=nn.ReLU()(out)
        return out

class SE_ResNet(pl.LightningModule):
    def __init__(self,depth,num_classes):
        super().__init__()
        block=SENet_block
        n=(depth-2)//6 # depth should be 6n+2

        self.inplanes=16
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(self.inplanes)
        self.relu=nn.ReLU(inplace=True)
        self.layer1=self._make_layer(block,16,n)
        self.layer2=self._make_layer(block,32,n,stride=2)
        self.layer3=self._make_layer(block,64,n,stride=2)
        #self.avgpool = nn.AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64*block.expansion,num_classes)

        # init paramater
        #for m in self.modules():
        #    if isinstance(m,nn.Conv2d):
        #        n = m.kernel_size[0]* m.kernel_size[1]*m.out_channels
        #        m.weight.data.normal_(0,math.sqrt(2./n))
        #    elif isinstance(m,nn.BatchNorm2d):
        #        m.weight.data.fill_(1)
        #        m.bias.data.zero_()

    def _make_layer(self,block,planes,blocks,stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes,planes*block.expansion,kernel_size=1,
                              stride=stride,bias=False),
                    nn.BatchNorm2d(planes*block.expansion)
            )
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))

        self.inplanes=planes*block.expansion
        for i in range(1,blocks):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        x=self.avgpool(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x
    
    def configure_optimizers(self):
        #return torch.optim.Adam(self.parameters(),lr=1e-3,weight_decay=0.01)
        return torch.optim.SGD(self.parameters(),lr=1e-1,momentum=0.1)

    def training_step(self,train_batch,train_idx):
        x,y=train_batch
        y_hat=self.forward(x)
        loss= F.cross_entropy(y_hat,y)
        self.log('Train loss (CrossEntropyLoss):', loss)
        return loss

    def validation_step(self,val_batch,train_idx):
        x,y=val_batch
        y_hat=self.forward(x)
        loss= F.cross_entropy(y_hat,y)
        self.log('Validation loss (CrossEntropyLoss):', loss)

def test_SENet_block():
    block=SENet_block(3,32).cuda()
    summary(block,input_size=(1, 3, 32, 32))
    a=torch.zeros(1,3,10,10).cuda()
    b=block(a)

def test_SE_ResNet():
    net=SE_ResNet(2+6*40,10).cuda()
    a=torch.zeros(1,3,100,100).cuda()
    b=net(a)
    summary(net,input_size=(1, 3, 150, 150))

def main():
    test_SE_ResNet()

if __name__ == '__main__':
    main()
