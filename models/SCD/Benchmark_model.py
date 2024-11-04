import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, is_downsample, stride=1, dilation=1, BatchNorm=nn.BatchNorm2d):
        super(ResBlock, self).__init__()
        # if norm_layer is None:
        #     norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

        self.is_downsample = is_downsample
        self.downsample = nn.Sequential(nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1),
                                        nn.BatchNorm2d(planes))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.is_downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HRSCD(nn.Module):
    def __init__(self, in_dim, out_dim, output_mid_feature=False):
        super(HRSCD, self).__init__()
        self.output_mid_feature = output_mid_feature
        self.conv_block_1 = nn.Sequential(
            ResBlock(inplanes=in_dim, planes=8, is_downsample=True),
            ResBlock(inplanes=8, planes=8, is_downsample=False)
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            ResBlock(inplanes=8, planes=16, is_downsample=True),
            ResBlock(inplanes=16, planes=16, is_downsample=False)
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            ResBlock(inplanes=16, planes=32, is_downsample=True),
            ResBlock(inplanes=32, planes=32, is_downsample=False),
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            ResBlock(inplanes=32, planes=64, is_downsample=True),
            ResBlock(inplanes=64, planes=64, is_downsample=False),
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_5 = nn.Sequential(
            ResBlock(inplanes=64, planes=128, is_downsample=True),
            ResBlock(inplanes=128, planes=128, is_downsample=False),
        )
        self.max_pool_5 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_6 = nn.Sequential(
            ResBlock(inplanes=128, planes=256, is_downsample=True),
            ResBlock(inplanes=256, planes=256, is_downsample=False),
        )


        self.conv_block_7 = nn.Sequential(
            ResBlock(inplanes=384, planes=64, is_downsample=True),
            ResBlock(inplanes=64, planes=64, is_downsample=False),
        )

        self.conv_block_8 = nn.Sequential(
            ResBlock(inplanes=128, planes=64, is_downsample=True),
            ResBlock(inplanes=64, planes=32, is_downsample=True),
        )

        self.conv_block_9 = nn.Sequential(
            ResBlock(inplanes=64, planes=32, is_downsample=True),
            ResBlock(inplanes=32, planes=16, is_downsample=True),
        )

        self.conv_block_10 = nn.Sequential(
            ResBlock(inplanes=32, planes=16, is_downsample=True),
            ResBlock(inplanes=16, planes=8, is_downsample=True),
        )

        self.conv_block_11 = nn.Sequential(
            ResBlock(inplanes=16, planes=16, is_downsample=False),
            ResBlock(inplanes=16, planes=16, is_downsample=False),
            nn.Conv2d(in_channels=16, out_channels=out_dim, kernel_size=1)
        )


    def forward(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        feature_5 = self.conv_block_5(down_feature_4)
        down_feature_5 = self.max_pool_5(feature_5)

        feature_6 = self.conv_block_6(down_feature_5)

        
        #####################
        # decoder
        #####################
        up_feature_6 = F.upsample_bilinear(feature_6, size=feature_5.size()[2:])
        concat_feature_7 = torch.cat([up_feature_6, feature_5], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_7 = F.upsample_bilinear(feature_7, size=feature_4.size()[2:])
        concat_feature_8 = torch.cat([up_feature_7, feature_4], dim=1)
        feature_8 = self.conv_block_8(concat_feature_8)

        up_feature_9 = F.upsample_bilinear(feature_8, size=feature_3.size()[2:])
        concat_feature_9 = torch.cat([up_feature_9, feature_3], dim=1)
        feature_9 = self.conv_block_9(concat_feature_9)

        up_feature_10 = F.upsample_bilinear(feature_9, size=feature_2.size()[2:])
        concat_feature_10 = torch.cat([up_feature_10, feature_2], dim=1)
        feature_10 = self.conv_block_10(concat_feature_10)

        up_feature_11 = F.upsample_bilinear(feature_10, size=feature_1.size()[2:])
        concat_feature_11 = torch.cat([up_feature_11, feature_1], dim=1)
        output = self.conv_block_11(concat_feature_11)

        if self.output_mid_feature:
            return feature_8, feature_9, feature_10, output
        else:
            return output


class HRSCD_S4(nn.Module):
    def __init__(self, in_dim_clf, in_dim_cd, out_dim_clf, out_dim_cd):
        super(HRSCD_S4, self).__init__()

        self.semantic_branch_1 = HRSCD(in_dim=in_dim_clf, out_dim=out_dim_clf, output_mid_feature=True)
        self.semantic_branch_2 = HRSCD(in_dim=in_dim_clf, out_dim=out_dim_clf, output_mid_feature=True)

        self.conv_block_1 = nn.Sequential(
            ResBlock(inplanes=in_dim_cd, planes=8, is_downsample=True),
            ResBlock(inplanes=8, planes=8, is_downsample=False)
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            ResBlock(inplanes=8, planes=16, is_downsample=True),
            ResBlock(inplanes=16, planes=16, is_downsample=False)
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            ResBlock(inplanes=16, planes=32, is_downsample=True),
            ResBlock(inplanes=32, planes=32, is_downsample=False),
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            ResBlock(inplanes=32, planes=64, is_downsample=True),
            ResBlock(inplanes=64, planes=64, is_downsample=False),
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_5 = nn.Sequential(
            ResBlock(inplanes=64, planes=128, is_downsample=True),
            ResBlock(inplanes=128, planes=128, is_downsample=False),
        )
        self.max_pool_5 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_6 = nn.Sequential(
            ResBlock(inplanes=128, planes=256, is_downsample=True),
            ResBlock(inplanes=256, planes=256, is_downsample=False),
        )


        self.conv_block_7 = nn.Sequential(
            ResBlock(inplanes=384, planes=64, is_downsample=True),
            ResBlock(inplanes=64, planes=64, is_downsample=False),
        )

        self.conv_block_8 = nn.Sequential(
            ResBlock(inplanes=192, planes=64, is_downsample=True),
            ResBlock(inplanes=64, planes=32, is_downsample=True),
        )

        self.conv_block_9 = nn.Sequential(
            ResBlock(inplanes=96, planes=32, is_downsample=True),
            ResBlock(inplanes=32, planes=16, is_downsample=True),
        )

        self.conv_block_10 = nn.Sequential(
            ResBlock(inplanes=48, planes=16, is_downsample=True),
            ResBlock(inplanes=16, planes=16, is_downsample=False),
        )

        self.conv_block_11 = nn.Sequential(
            ResBlock(inplanes=24, planes=16, is_downsample=True),
            ResBlock(inplanes=16, planes=16, is_downsample=False),
            nn.Conv2d(in_channels=16, out_channels=out_dim_cd, kernel_size=1)
        )

    def forward(self, map_data, satellite_img, concat_data):
        land_cover_feature_11, land_cover_feature_12, land_cover_feature_13, land_cover_clf_1 = self.semantic_branch_1(
            map_data)
        land_cover_feature_21, land_cover_feature_22, land_cover_feature_23, land_cover_clf_2 = self.semantic_branch_2(
            satellite_img)
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(concat_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        feature_5 = self.conv_block_5(down_feature_4)
        down_feature_5 = self.max_pool_5(feature_5)

        feature_6 = self.conv_block_6(down_feature_5)
        
        #####################
        # decoder
        #####################
        up_feature_6 = F.upsample_bilinear(feature_6, size=feature_5.size()[2:])
        concat_feature_7 = torch.cat([up_feature_6, feature_5], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_7 = F.upsample_bilinear(feature_7, size=feature_4.size()[2:])
        concat_feature_8 = torch.cat([up_feature_7, feature_4, land_cover_feature_11, land_cover_feature_21], dim=1)
        feature_8 = self.conv_block_8(concat_feature_8)

        up_feature_9 = F.upsample_bilinear(feature_8, size=feature_3.size()[2:])
        concat_feature_9 = torch.cat([up_feature_9, feature_3, land_cover_feature_12, land_cover_feature_22], dim=1)
        feature_9 = self.conv_block_9(concat_feature_9)

        up_feature_10 = F.upsample_bilinear(feature_9, size=feature_2.size()[2:])
        concat_feature_10 = torch.cat([up_feature_10, feature_2, land_cover_feature_13, land_cover_feature_23], dim=1)
        feature_10 = self.conv_block_10(concat_feature_10)

        up_feature_11 = F.upsample_bilinear(feature_10, size=feature_1.size()[2:])
        concat_feature_11 = torch.cat([up_feature_11, feature_1], dim=1)
        cd_clf = self.conv_block_11(concat_feature_11)

        return cd_clf, land_cover_clf_1, land_cover_clf_2
