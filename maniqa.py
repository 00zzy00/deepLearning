import torch
import torch.nn as nn
import timm
from timm.models.resnet import resnet50

from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange


class TABlock(nn.Module):
    """注意力模块"""

    def __init__(self, dim, drop=0.1):
        super().__init__()
        # “dim” 表示输入特征的维度大小。
        # 在这个自注意力机制模块中，输入 x 的维度是 (B, C, N)，
        # 其中 B 是batch size，C 是输入特征的通道数，N 是特征的长度（可以理解为序列的长度）
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        """计算注意力得分时的归一化因子。归一化因子被定义为输入特征维度 dim 的倒数的平方根
        作用：对不同特征维度的输入具有更好的泛化能力"""
        self.norm_fact = dim ** -0.5
        """dim=-1 表示对最后一个维度进行 Softmax 计算,
        将每个输入样本沿着最后一个维度进行归一化"""
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        """构造器，该类用于保存模型输出SaveOutput，该类包括属性：outputs列表"""
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        """当调用模块时，将输出添加入outputs列表"""
        self.outputs.append(module_out)

    def clear(self):
        """用于清空outputs列表"""
        self.outputs = []


class MANIQA(nn.Module):
    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        """//===>除完之后的整数部分"""
        self.input_size = img_size // patch_size
        """得到每一个patch的宽高"""
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)
        """建立预训练的VIT，patch_size = 8,image_size=224"""
        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        # """利用模型工厂函数完成，resnet模型的建立"""
        # self.resnet50 = timm.create_model('resnet50', pretrained=True)
        """实例化一个SaveOutput对象"""
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )
        """多分支结构"""
        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )


    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x):
        # 使用 Vision Transformer (self.vit) 对输入 x 进行前向传播，结果存储在 _x 中。
        _x = self.vit(x)
        # 调用了带有 save_output 实例作为参数的 extract_feature 方法。该方法从保存的输出中提取特征。
        x = self.extract_feature(self.save_output)
        # 这样做是为了在下一次前向传播时重置保存的输出。
        self.save_output.outputs.clear()
        print(f"ViT output shape: {_x.shape}")
        print(f"feature shape:{x.shape}")
        return _x,x

        # stage 1
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        score = torch.tensor([]).cuda()
        for i in range(x.shape[0]):
            f = self.fc_score(x[i])
            w = self.fc_weight(x[i])
            _s = torch.sum(f * w) / torch.sum(w)
            score = torch.cat((score, _s.unsqueeze(0)), 0)
        return score

if __name__ == '__main__':
    model = MANIQA()
    dummy_input = torch.randn(1,3,224,224)
    with torch.no_grad():
        model.eval()
        output = model(dummy_input)