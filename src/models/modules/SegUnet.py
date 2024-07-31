# Taken from https://github.com/AntanasKascenas/DenoisingAE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from kornia.losses import FocalLoss
from src.models.modules.DAE_UNET import UNet, UNetConvBlock, UNetUpBlock, get_groups, sqrt
import monai
import copy
import lpips
class SegUnet(nn.Module):
    """SiamUnet segmentation network."""

    def __init__(
        self,
        cfg,
        in_channels=1,
        n_classes=1,
        depth=4,
        wf=6,
        padding=True,
        norm="group",
        up_mode='upconv'):
        """
        A modified U-Net implementation [1].
        [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
            Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            norm (str): one of 'batch' and 'group'.
                        'batch' will use BatchNormalization.
                        'group' will use GroupNormalization.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(SegUnet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        fac = 2 # account for concatenated features in the decoder
        self.cfg = cfg
        self.deepsup = cfg.get('deepsup',False)
        self.down_path = nn.ModuleList()
        self.cond_path_down = nn.ModuleList()
        self.cond_path_up = nn.ModuleList()
        if self.deepsup:
            self.deepsup_1x1 = nn.ModuleList()

        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
                
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            
            self.up_path.append(
                UNetUpBlock(prev_channels * fac, 2 ** (wf + i) * fac, up_mode, padding, norm=norm)
            )
            if self.deepsup:
                self.deepsup_1x1.append(nn.Conv2d(2 ** (wf + i) * fac, 1, kernel_size=1))
            prev_channels = 2 ** (wf + i)

        self.imageDim = [int(x/cfg.rescaleFactor) for x in cfg.imageDim] if not cfg.get('patched_processing',False) else tuple([cfg.get('proc_patch_size',16)]*3) # Input Dimension of the Image

        self.last = nn.Conv2d(prev_channels * fac, n_classes, kernel_size=1)
        self.pool = nn.AvgPool2d(self.imageDim[0]) # for feature exraction
        self.sm = nn.Sigmoid()



        if cfg.get('loss_siam','bce') == 'bce':
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        elif cfg.get('loss_siam','bce') == 'dice':
            self.loss = monai.losses.DiceLoss(sigmoid=True)
        else:
            raise NotImplementedError

    def forward_down(self, x, nosharing=False):
        blocks = []
        if nosharing:
            for i, down in enumerate(self.down_path2):
                x = down(x)
                blocks.append(x)
                if i != len(self.down_path2) - 1:
                    x = F.avg_pool2d(x, 2)
        else:
            for i, down in enumerate(self.down_path):
                x = down(x)
                blocks.append(x)
                if i != len(self.down_path) - 1:
                    x = F.avg_pool2d(x, 2)

        return x, blocks

    def forward_up_without_last(self, x, blocks,deepsup=False):
        if deepsup:
            dec_blocks = []
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)
            if deepsup:
                dec_blocks.append(x)
        if deepsup:
            return x, dec_blocks
        else:
            return x

    def forward_without_last(self, x,deepsup=False):
        x, blocks = self.forward_down(x)


        if deepsup:
            x, blocks_out = self.forward_up_without_last(x, blocks,deepsup=deepsup)
            return x, blocks_out
        else:
            x = self.forward_up_without_last(x, blocks,deepsup=deepsup)
            return x
        
    def upsample(self, in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
        in_H, in_W = in_tens.shape[2], in_tens.shape[3]
        return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)
    def get_features(self, x):
        return self.forward_without_last(x)

    def forward_siamese(self, x1, x2,deepsup=False): 
        x1 = x2 - x1 if not self.cfg.get('l2_res',False) else (x2 - x1)**2
        z1, blocks_1 = self.forward_down(x1) # residual
        z2, blocks_2 = self.forward_down(x2) # potentially abnormal input
        z = torch.cat([z1,z2],dim=1) # concatenate the features
        blocks = [torch.cat([b1,b2],dim=1) for b1,b2 in zip(blocks_1,blocks_2)] # concatenate the features

        if deepsup:
            z, blocks_out = self.forward_up_without_last(z, blocks,deepsup=deepsup)
            return self.last(z), blocks_out
        else: 
            z = self.forward_up_without_last(z, blocks,deepsup=deepsup)
            return self.last(z)
        

    def score_diff(self, x1, x2, x2_anomal, anomaly_map=None,deepsup=False): # x1: reconstruction, x2: target, anomaly_x2: target with anomaly
        if deepsup:
            x, blocks_out = self.forward_siamese(x1, x2_anomal,deepsup=deepsup)
            # perform deep supervision
            results = self.loss(x, anomaly_map)
            for idx, b in enumerate(blocks_out):
                # 1x1 conv to single channgel
                b_ = self.deepsup_1x1[idx](b)
                # rescale anomaly map to match the size of the block
                anomaly_map_rescaled = F.interpolate(anomaly_map, size=b_.shape[-2:])
                results += self.loss(b_, anomaly_map_rescaled)
        else:
            x = self.forward_siamese(x1, x2_anomal)

        if anomaly_map is not None: # during training
            if deepsup: 
                results = results
            else:
                results = self.loss(x, anomaly_map)
        else: 
            results = self.sm(x)
        return results
    
