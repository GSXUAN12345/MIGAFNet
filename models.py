import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from mamba_ssm import Mamba
except ImportError:
    print("Error: mamba_ssm library not found.")
    class Mamba(nn.Module):
        def __init__(self, **kwargs): super().__init__()
        def forward(self, x): return x

class HSI_FrontEnd(nn.Module):
    def __init__(self, in_bands, out_channels=64, d_state=16):
        super().__init__()
        self.spectral_mamba = Mamba(
            d_model=1,      
            d_state=d_state, 
            d_conv=4, 
            expand=2
        )
        self.norm_mamba = nn.LayerNorm(1)
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True)
        )
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=8 * in_bands, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, N, H, W = x.shape
        x_seq = rearrange(x, 'b n h w -> (b h w) n 1')
        residual = x_seq
        x_seq = self.norm_mamba(x_seq)
        x_mamba = self.spectral_mamba(x_seq) + residual 
        x_3d = rearrange(x_mamba, '(b h w) n 1 -> b 1 n h w', b=B, h=H, w=W)
        x_conv3d = self.conv3d(x_3d) 
        x_flat = rearrange(x_conv3d, 'b c n h w -> b (c n) h w')
        out_hsi = self.projection(x_flat) 
        
        return out_hsi

class LiDAR_FrontEnd(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class VisualMambaBlock(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_bwd = Mamba(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        residual = x_flat
        x_norm = self.norm(x_flat)
        out_fwd = self.mamba_fwd(x_norm)
        x_reverse = x_norm.flip([1])
        out_bwd = self.mamba_bwd(x_reverse)
        out_bwd = out_bwd.flip([1])
        out = out_fwd + out_bwd
        out = out + residual
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        return out
        
class DeepIGAF(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.gate_h2l = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.gate_l2h = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.Sigmoid()
        )
        self.proj_h = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), 
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.proj_l = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim), 
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x_hsi, x_lidar):
        gate_h = self.gate_l2h(x_lidar) 
        gate_l = self.gate_h2l(x_hsi)   
        hsi_out = x_hsi * gate_h + x_hsi 
        lidar_out = x_lidar * gate_l + x_lidar
        hsi_out = self.proj_h(hsi_out)
        lidar_out = self.proj_l(lidar_out)
        
        return hsi_out, lidar_out

class FusionBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.mamba_h = VisualMambaBlock(dim)
        self.mamba_l = VisualMambaBlock(dim)
        self.igaf = DeepIGAF(dim)
        
    def forward(self, x_h, x_l):
        x_h = self.mamba_h(x_h)
        x_l = self.mamba_l(x_l)
        x_h, x_l = self.igaf(x_h, x_l)
        
        return x_h, x_l

class MIGAFNet(nn.Module):
    def __init__(self, hsi_channels, lidar_channels, n_classes, patch_size):
        super().__init__()
        self.dim = 64 
        self.num_layers = 3 
        
        self.hsi_stem = HSI_FrontEnd(in_bands=hsi_channels, out_channels=self.dim)
        self.lidar_stem = LiDAR_FrontEnd(in_channels=lidar_channels, out_channels=self.dim)
        self.layers = nn.ModuleList([
            FusionBlock(self.dim) for _ in range(self.num_layers)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(128, n_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_hsi, x_lidar):
        feat_h = self.hsi_stem(x_hsi)   
        feat_l = self.lidar_stem(x_lidar) 
        for layer in self.layers:
            feat_h, feat_l = layer(feat_h, feat_l)
        feat_final = torch.cat([feat_h, feat_l], dim=1) 
        feat_pool = self.pool(feat_final).flatten(1)    
        out = self.classifier(feat_pool)
        
        return out