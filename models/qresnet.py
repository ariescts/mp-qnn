import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.utils import load_state_dict_from_url

from .quantization import Clamp, Hardtanh6, QConv2d, QLinear, OPTA
assert OPTA in ('round', 'sign')


__all__ = ['QResNet', 'qresnet18', 'qresnet34', 'qresnet50', 'qresnet101']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def qconv3x3(ka, kw, in_planes, out_planes, stride=1, groups=1, dilation=1):
    return QConv2d(ka, kw, in_planes, out_planes, kernel_size=3, padding=dilation, stride=stride, groups=groups,
                   dilation=dilation, bias=False, opta=OPTA)


def qconv1x1(ka, kw, in_planes, out_planes, stride=1):
    return QConv2d(ka, kw, in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ka, kw, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None, first=False, last=False,):
        super().__init__()
        self.first = first
        self.last = last
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')

        if first:
            self.conv1 = qconv3x3(32, kw, inplanes, planes, stride)
        else:
            self.conv1 = qconv3x3(ka, kw, inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.conv2 = qconv3x3(ka, kw, planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        # self.clamp = nn.Hardtanh(min_val=0, max_val=1, inplace=True)
        if ka == 32:
            self.act = nn.ReLU(inplace=True)
        elif ka == 1 and OPTA == 'sign':
            # self.act = nn.Hardtanh(min_val=-1, max_val=1, inplace=True)
            self.act = Hardtanh6()
        else:
            # self.act = nn.Hardtanh(min_val=0, max_val=1, inplace=True)
            self.act = Clamp()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out) if self.last else self.act(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ka, kw, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, first=False, last=False):
        super().__init__()
        self.first =first
        self.last = last
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width/64.)) * groups

        if first:
            self.conv1 = qconv1x1(32, kw, inplanes, width)
        else:
            self.conv1 = qconv1x1(ka, kw, inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = qconv3x3(ka, kw, width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = qconv1x1(ka, kw, width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
        # self.clamp = nn.Hardtanh(min_val=0, max_val=1, inplace=True)
        if ka == 32:
            self.act = nn.ReLU(inplace=True)
        elif ka == 1 and OPTA == 'sign':
            # self.act = nn.Hardtanh(min_val=-1, max_val=1, inplace=True)
            self.act = Hardtanh6()
        else:
            # self.act = nn.Hardtanh(min_val=0, max_val=1, inplace=True)
            self.act = Clamp()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = x if self.downsample is None else self.downsample(x)
        out += identity
        out = self.relu(out) if self.last else self.act(out)

        return out


class QResNet(nn.Module):
    def __init__(self, block, layers, ka=[32, 4, 4, 4, 4], kw=[32, 4, 4, 4, 4], num_classes=1000, groups=1,
                 zero_init_residual=False, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()
        self.ka = ka
        self.kw = kw
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None or a 3-element tuple, '
                             'got {}'.format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if kw[0] == 32:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.act = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(ka[1], kw[1], block, 64, layers[0], first=True)
        else:
            self.conv1 = QConv2d(32, kw[0], 3, self.inplanes, kernel_size=7, padding=3, stride=2)
            self.bn1 = norm_layer(self.inplanes)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.act = Clamp()
            self.layer1 = self._make_layer(ka[1], kw[1], block, 64, layers[0], first=False)
        self.layer2 = self._make_layer(ka[2], kw[2], block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(ka[3], kw[3], block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(ka[4], kw[4], block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], last=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, ka, kw, block, planes, blocks, stride=1, dilate=False, first=False, last=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(
                qconv1x1(32, 32, self.inplanes, planes*block.expansion, stride),
                norm_layer(planes*block.expansion),
            )

        layers = []
        layers.append(block(ka, kw, self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            previous_dilation, norm_layer, first=first))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks-1):
            layers.appen(block(ka, kw, self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                               dilation=self.dilation, norm_layer=norm_layer))
        layers.append(block(ka, kw, self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                            dilation=self.dilation, norm_layer=norm_layer, last=last))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.act(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _qresnet(arch, block, layers, ka, kw, pretrained, progress, **kwargs):
    model = QResNet(block, layers, ka, kw, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def qresnet18(ka=[32, 4, 4, 4, 4], kw=[32, 4, 4, 4, 4], pretrained=True, progress=True, **kwargs):
    return _qresnet('resnet18', BasicBlock, [2, 2, 2, 2], ka, kw, pretrained, progress, **kwargs)


def qresnet34(ka=[32, 4, 4, 4, 4], kw=[32, 4, 4, 4, 4], pretrained=False, progress=True, **kwargs):
    return _qresnet('resnet34', BasicBlock, [3, 4, 6, 3], ka, kw, pretrained, progress, **kwargs)


def qresnet50(ka=[32, 4, 4, 4, 4], kw=[32, 4, 4, 4, 4], pretrained=False, progress=True, **kwargs):
    return _qresnet('resnet50', Bottleneck, [3, 4, 6, 3], ka, kw, pretrained, progress, **kwargs)


def qresnet101(ka=[32, 4, 4, 4, 4], kw=[32, 4, 4, 4, 4], pretrained=False, progress=True, **kwargs):
    return _qresnet('resnet101', Bottleneck, [3, 4, 23, 3], ka, kw, pretrained, progress, **kwargs)



if __name__ == '__main__':
    model1 = qresnet18(ka=[32, 4, 4, 4, 4], kw=[32, 4, 4, 4, 4], pretrained=True)
    model2 = qresnet18(ka=[8, 4, 4, 4, 4], kw=[8, 4, 4, 4, 4], pretrained=True)
    x = torch.randn(2, 3, 224, 224)
    y1 = model1(x)
    print(y1.shape)
    y2 = model2(x)
    print(y2.shape)
