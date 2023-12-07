import torch
from torch import nn
import torchvision
import torchvision.models
import torch.nn.functional as F

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


class BaseRN50(nn.Module):
    def __init__(self):
        super(BaseRN50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        bs, ts, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        x = self.model(x)
        return x.reshape(bs, ts, 1000)


class BaseVGG11(nn.Module):
    def __init__(self, pretrained):
        super(BaseVGG11, self).__init__()
        self.model = torchvision.models.vgg11_bn(pretrained=pretrained)

    def forward(self, x):
        return self.model.features(x)


class Embedder(nn.Module):
    def __init__(self, args):
        super(Embedder, self).__init__()
        self.args = args
        self.num_context_steps = args.num_context_steps

        if args.base_model_name == 'resnet50':
            self.base_model = resnet50(pretrained=True)
            del self.base_model.layer4
            del self.base_model.fc
        elif args.base_model_name == 'resnet18':
            self.base_model = resnet18(pretrained=False)
            del self.base_model.layer4
            del self.base_model.fc
        elif args.base_model_name == 'vgg11':
            self.base_model = BaseVGG11(pretrained=False)
        else:
            raise NotImplementedError

        if args.freeze_base:
            self.freeze_base_model()

        c = 1024 if 'resnet' in args.base_model_name else 512
        self.conv_layers = nn.Sequential(
            nn.Conv3d(c, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        if args.input_size == 224:
            self.ksize = 14
        elif args.input_size == 168:
            self.ksize = 11 if 'resnet' in args.base_model_name else 5
        else:
            raise NotImplementedError

        self.maxpool = nn.MaxPool3d(
            (self.num_context_steps, self.ksize, self.ksize)
        )
        self.embedding_layer = nn.Linear(256, args.embedding_size)
        self.dropout = nn.Dropout(0.1)

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x, bbox=None):
        # x: (bs, ts=32, 3, 224/168, 224/168)
        bs, ts, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        x = self.base_model(x)  # (bs, 3, 168, 168) -> (bs, 1024, 11, 11)

        _, c, h, w = x.size()
        x = x.contiguous().view(-1, c, self.num_context_steps, h, w)
        x = self.dropout(x)
        x = self.conv_layers(x)
        x = self.maxpool(x)

        _, c, _, _, _ = x.size()
        x = x.reshape(bs, -1, c)
        x = self.fc_layers(x)
        x = self.embedding_layer(x.contiguous())
        return x


class RoIPosEmbedder(Embedder):
    def __init__(self, args):
        super(RoIPosEmbedder, self).__init__(args)
        self.roi_output_size = 4
        self.n_boxes = 4  # 2 hand + 2 object
        self.n_tokens = self.n_boxes + 1  # 4 local + 1 global
        self.dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.n_heads = 4
        self.dp_rate = 0.1
        self.use_mask = args.use_mask
        self.use_bbox_pe = args.use_bbox_pe
        self.weigh_token_by_bbox = args.weigh_token_by_bbox

        self.maxpool_context = nn.MaxPool3d(
            (self.num_context_steps, 1, 1)
        )
        self.roi_align = torchvision.ops.RoIAlign(output_size=(self.roi_output_size, self.roi_output_size),
                                                  spatial_scale=0.25,
                                                  sampling_ratio=-1)
        self.proj_local = nn.Linear(64 * self.roi_output_size * self.roi_output_size, self.dim)
        self.proj_global = nn.Linear(256, self.dim)

        self.scale_factors = torch.tensor([1 / self.args.input_size] * 4 + [1], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.randn(5, self.dim), requires_grad=True)
        self.token_embed = nn.Parameter(torch.randn(1, self.n_tokens, self.dim), requires_grad=True)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.n_heads, dropout=self.dp_rate,
                                                     batch_first=True),
            num_layers=self.n_layers
        )
        self.ln = nn.LayerNorm(self.dim)
        self.embedding_layer = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, args.embedding_size)
        )

    def pool_context_frames(self, x):
        _, c, h, w = x.size()
        x = x.contiguous().view(-1, c, self.num_context_steps, h, w)
        x = self.dropout(x)
        return self.maxpool_context(x).squeeze()

    def forward(self, x, bbox):
        bs, ts, c, h, w = x.size()
        x = x.reshape(-1, c, h, w)
        bbox = bbox.reshape(-1, bbox.shape[2], bbox.shape[3])
        bbox_list = torch.chunk(bbox[:, :, 0:4], chunks=bbox.shape[0], dim=0)
        bbox_list = [b.squeeze() for b in bbox_list]
        bbox_id = bbox[:, :, -1].long()

        x_mid, x = self.base_model(x, middle=True)  # (bs, 3, 168, 168) -> (bs, 256, 42, 42) -> (bs, 1024, 11, 11)

        x_mid = self.pool_context_frames(x_mid)     # (bs*ts, 64, 56, 56)
        x_roi = self.roi_align(x_mid, bbox_list)    # (bs*ts*4, 64, 4, 4)
        x_roi = x_roi.reshape(-1, self.n_boxes, *(x_roi.shape[1:]))  # (bs, ts, 64, 4, 4)
        x_local = torch.flatten(x_roi, start_dim=2)  # (bs, ts, 1024)
        x_local = self.proj_local(x_local)

        if self.use_bbox_pe:
            bbox_pe = bbox[:, :, 0:5] * self.scale_factors.to(bbox.device)
            local_pe = torch.matmul(bbox_pe, self.pos_embed)
            x_local = x_local + local_pe
        x_local = x_local + self.token_embed[0][bbox_id]
        if self.weigh_token_by_bbox:
            bbox_prob = bbox[:, :, 4].unsqueeze(-1)
            x_local = x_local * bbox_prob

        _, c, h, w = x.size()
        x = x.contiguous().view(-1, c, self.num_context_steps, h, w)
        x = self.dropout(x)
        x = self.conv_layers(x)
        x = self.maxpool(x)
        x_global = self.proj_global(x.squeeze().unsqueeze(1))
        x_global = x_global + self.token_embed[:, -1, :]

        x = torch.cat((x_local, x_global), dim=1)  # (bs*ts/2, n_tokens, dim)
        x = self.ln(x)
        if self.use_mask:
            mask_local = bbox_id.eq(-1)
            mask_global = torch.full((mask_local.size(0), 1), False, dtype=torch.bool, device=mask_local.device) # do not mask
            mask = torch.cat([mask_local, mask_global], dim=1)
            output = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            output = self.transformer_encoder(x)
        y = output.mean(dim=1)
        y = self.embedding_layer(y)
        return y.reshape(bs, -1, y.shape[-1])


'''Adopt from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation>1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3
    # convolution(self.conv2) while original implementation places the stride
    # at the first 1x1 convolution(self.conv1) according to "Deep residual
    # learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy
    # according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(f"replace_stride_with_dilation should be None or"
                             f"a 3-element tuple, got "
                             f"{replace_stride_with_dilation}")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual
        # block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, middle):
        # See note [TorchScript super()]
        x = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        if middle:
            return x, x3
        # NOTE: temporal comment out to use the model for feature extraction
        # x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x3

    def forward(self, x, middle=False):
        return self._forward_impl(x, middle)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', Bottleneck, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition"
    https://arxiv.org/pdf/1512.03385.pdf`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to
            stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

