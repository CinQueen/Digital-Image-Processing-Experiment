import torch
import torch.nn as nn
from models.mobilenet_v2 import MobileNetV2

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNMobileNet(nn.Module):
    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super().__init__()
        
        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer=norm_layer, pretrained=pretrained)

        # 保持原始head结构
        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        # 保持原始smooth层结构
        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_filters),
            nn.LeakyReLU(0.2)
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_filters // 2),
            nn.LeakyReLU(0.2)
        )

        # 修改最终输出层，但保持通道数不变
        self.final = nn.Sequential(
            nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # 添加特征裁剪层
        self.feature_clip = lambda x: torch.clamp(x, -10, 10)  # 限制特征范围

    def forward(self, x):
        # 添加值域检查和打印
        print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
        
        map0, map1, map2, map3, map4 = self.fpn(x)
        
        # 特征稳定化
        def stabilize_features(feat):
            # 限制范围
            feat = torch.clamp(feat, -10, 10)
            # 去除异常值
            mean = torch.mean(feat, dim=[2,3], keepdim=True)
            std = torch.std(feat, dim=[2,3], keepdim=True)
            feat = torch.clamp(feat, mean - 3*std, mean + 3*std)
            # 归一化
            return (feat - mean) / (std + 1e-8)
        
        map0 = stabilize_features(map0)
        map1 = stabilize_features(map1)
        map2 = stabilize_features(map2)
        map3 = stabilize_features(map3)
        map4 = stabilize_features(map4)
        
        # 使用双线性插值进行上采样
        interpolate_mode = "bilinear"
        align_corners = True
        
        # Head处理
        map4 = self.head4(map4)
        map3 = self.head3(map3)
        map2 = self.head2(map2)
        map1 = self.head1(map1)
        
        # 特征融合
        map4 = nn.functional.interpolate(map4, size=map1.shape[-2:], mode=interpolate_mode, align_corners=align_corners)
        map3 = nn.functional.interpolate(map3, size=map1.shape[-2:], mode=interpolate_mode, align_corners=align_corners)
        map2 = nn.functional.interpolate(map2, size=map1.shape[-2:], mode=interpolate_mode, align_corners=align_corners)
        
        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.interpolate(smoothed, size=map0.shape[-2:], mode=interpolate_mode, align_corners=align_corners)
        smoothed = self.smooth2(smoothed + map0)
        
        # 最终处理
        smoothed = nn.functional.interpolate(smoothed, size=x.shape[-2:], mode=interpolate_mode, align_corners=align_corners)
        output = self.final(smoothed)
        
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        return output


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=True):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """

        super().__init__()
        net = MobileNetV2(n_class=1000)
        self.features = net.features

        self.enc0 = nn.Sequential(*self.features[0:2])
        self.enc1 = nn.Sequential(*self.features[2:4])
        self.enc2 = nn.Sequential(*self.features[4:7])
        self.enc3 = nn.Sequential(*self.features[7:11])
        self.enc4 = nn.Sequential(*self.features[11:16])

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        self.lateral4 = nn.Conv2d(160, num_filters, kernel_size=1, bias=False)
        self.lateral3 = nn.Conv2d(64, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(32, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(24, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(16, num_filters // 2, kernel_size=1, bias=False)

        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True


    def forward(self, x):
        # Bottom-up pathway
        enc0 = self.enc0(x)
        print(f"[FPN] enc0: shape={enc0.shape}, range=[{enc0.min():.3f}, {enc0.max():.3f}]")
        
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        # Lateral connections
        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)
        
        print(f"[FPN] Lateral connections:")
        print(f"lateral4: shape={lateral4.shape}, range=[{lateral4.min():.3f}, {lateral4.max():.3f}]")
        print(f"lateral0: shape={lateral0.shape}, range=[{lateral0.min():.3f}, {lateral0.max():.3f}]")
        
        # Top-down pathway
        map4 = lateral4
        map3 = self.td1(lateral3 + nn.functional.interpolate(
            map4, size=lateral3.shape[-2:], mode="bilinear", align_corners=False
        ))
        map2 = self.td2(lateral2 + nn.functional.interpolate(
            map3, size=lateral2.shape[-2:], mode="bilinear", align_corners=False
        ))
        map1 = self.td3(lateral1 + nn.functional.interpolate(
            map2, size=lateral1.shape[-2:], mode="bilinear", align_corners=False
        ))
        
        return lateral0, map1, map2, map3, map4

