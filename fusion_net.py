import torch
import torch.nn as nn
# ✅ 修改点1：导入 rsnet18
from drsncww import rsnet18 

class FusionNet(nn.Module):
    def __init__(self, num_classes=6):
        super(FusionNet, self).__init__()
        
        # === 左眼：RD 分支 ===
        # ✅ 修改点2：使用 rsnet18
        self.branch_rd = rsnet18(input_channels=1)
        self.branch_rd.fc = nn.Identity() 
        self.dim_rd = 512 * 1 
        
        # === 右眼：Track 分支 ===
        # ✅ 修改点3：使用 rsnet18
        self.branch_track = rsnet18(input_channels=2)
        self.branch_track.fc = nn.Identity()
        self.dim_track = 512 * 1
        
        # === 融合头 ===
        self.classifier = nn.Sequential(
            nn.Linear(self.dim_rd + self.dim_track, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_rd, x_track):
        feat_rd = self.branch_rd(x_rd)
        feat_track = self.branch_track(x_track)
        combined = torch.cat((feat_rd, feat_track), dim=1)
        out = self.classifier(combined)
        return out