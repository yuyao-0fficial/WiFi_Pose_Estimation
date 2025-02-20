# author: Schwarzer_land

import torch
from torch import nn
from torch.nn import functional as F


# 卷积层
class ConvGRUCell(nn.Module):

    def __init__(self, input_dim, output_dim, kernel_size, bias=True):
        """
        Initialize ConvGRU cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        output_dim: int
            Number of channels of output channel.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether to add the bias, True in default.
        """

        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        gru_input_dim = input_dim + output_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2  # 保证在传递过程中 （h,w）不变
        self.bias = bias

        # filters used for gates
        # r and c are calculated together
        self.gate_conv = nn.Conv2d(gru_input_dim,
                                   self.output_dim * 2,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   bias=self.bias)
        self.reset_gate_norm = nn.GroupNorm(1, self.output_dim, 1e-6, True)
        self.update_gate_norm = nn.GroupNorm(1, self.output_dim, 1e-6, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_dim,
                                     self.output_dim,
                                     kernel_size=self.kernel_size,
                                     padding=self.padding,
                                     bias=self.bias)
        self.output_norm = nn.GroupNorm(1, self.output_dim, 1e-6, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        cat_input_1 = torch.cat((x, h), dim=1)
        gated_input = self.gate_conv(cat_input_1)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        dim = gated_input.shape[1]
        r, z = torch.split(gated_input, dim // 2, 1)

        r_normed = self.reset_gate_norm(r)
        z_normed = self.update_gate_norm(z)
        r_normed_gated = torch.sigmoid(r_normed)
        z_normed_gated = torch.sigmoid(z_normed)
        return r_normed_gated, z_normed_gated

    def output(self, x, h, r):
        cat_input_2 = torch.cat((x, r * h), dim=1)
        out = self.output_conv(cat_input_2)
        out_normed = self.output_norm(out)
        out_normed_act = self.activation(out_normed)
        return out_normed_act

    def forward(self, input_tensor, h_cur):
        batch_size, channel, high, width = input_tensor.shape
        if h_cur is None:
            h_cur = torch.zeros((batch_size, self.output_dim, high, width), dtype=torch.float, device=input_tensor.device)
        r, z = self.gates(input_tensor, h_cur)
        h_hat = self.output(input_tensor, h_cur, r)
        h_next = (1 - z) * h_hat + z * h_hat

        return h_next

    def init_hidden(self, batch_size, image_size):
        """xl
        初始状态张量初始化.第一个timestamp的状态张量0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        init_h = torch.zeros(batch_size, self.output_dim, height, width, device=self.conv.weight.device)
        return init_h


class StackedConvGRU(nn.Module):
    """
    Parameters:参数介绍
        input_dim: Number of channels in input# 输入张量的通道数
        hidden_dim: Number of hidden channels # h,c两个状态张量的通道数，可以是一个列表
        kernel_size: Size of kernel in convolutions # 卷积核的尺寸，默认所有层的卷积核尺寸都是一样的,也可以设定不通lstm层的卷积核尺寸不同
        num_layers: Number of LSTM layers stacked on each other # 卷积层的层数，需要与len(hidden_dim)相等
        batch_first: Whether dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers # 是否返回所有lstm层的h状态
        Note: Will do same padding. # 相同的卷积核尺寸，相同的padding尺寸
    Input:输入介绍
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]# 需要是5维的
    Output:输出介绍
        返回的是两个列表：layer_output_list，last_state_list
        列表0：layer_output_list--单层列表，每个元素表示一层LSTM层的输出h状态,每个元素的size=[B,T,hidden_dim,H,W]
        列表1：last_state_list--双层列表，每个元素是一个二元列表[h,c],表示每一层的最后一个timestamp的输出状态[h,c],h.size=c.size = [B,hidden_dim,H,W]
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:使用示例
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, output_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(StackedConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)  # 转为列表
        output_dim = self._extend_for_multilayer(output_dim, num_layers)  # 转为列表
        if not len(kernel_size) == len(output_dim) == num_layers:  # 判断一致性
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):  # 多层GRU设置
            # 当前GRU层的输入维度
            # if i==0:
            #     cur_input_dim = self.input_dim
            # else:
            #     cur_input_dim = self.hidden_dim[i - 1]
            cur_input_dim = self.input_dim if i == 0 else self.output_dim[i - 1]  # 与上等价
            cell_list.append(ConvGRUCell(input_dim=cur_input_dim,
                                         output_dim=self.output_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)  # 把定义的多个LSTM层串联成网络模型

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvGRU
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            b, _, _, h, w = input_tensor.size()  # 自动获取 b,h,w信息
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)  # 根据输入张量获取gru的长度
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):  # 逐层计算

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):  # 逐个stamp计算
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=h)
                output_inner.append(h)  # 第 layer_idx 层的第t个stamp的输出状态

            layer_output = torch.stack(output_inner, dim=1)  # 第 layer_idx 层的第所有stamp的输出状态串联
            cur_layer_input = layer_output  # 准备第layer_idx+1层的输入张量

            layer_output_list.append(layer_output)  # 当前层的所有timestamp的h状态的串联
            last_state_list.append(h)  # 当前层的最后一个stamp的输出状态的h

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        所有lstm层的第一个timestamp的输入状态0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        检测输入的kernel_size是否符合要求，要求kernel_size的格式是list或tuple
        :param kernel_size:
        :return:
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        扩展到多层lstm情况
        :param param:
        :param num_layers:
        :return:
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


# 卷积层
def conv(in_channels, out_channels, kernlsize=3, pad=1, if2d=True):
    """
    :param in_channels: 输入通道数
    :param out_channels: 卷积后输出的通道数
    :param if2d: Whether the dimension of convolution is 2d, bool
    :return:
    """
    layer = []
    if if2d:
        layer.append(nn.BatchNorm2d(in_channels))
        layer.append(nn.LeakyReLU(inplace=False))
        layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernlsize, padding=pad, bias=True))
    else:
        layer.append(nn.BatchNorm1d(in_channels))
        layer.append(nn.LeakyReLU(inplace=False))
        layer.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernlsize, padding=pad, bias=True))
    return nn.Sequential(*layer)


# 上采样+拼接
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, if2d=True):
        """
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param if2d: Whether the dimension of convolution is 2d, bool
        """
        super(Up, self).__init__()
        # 转置卷积实现上采样
        if if2d:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = conv(in_channels, out_channels, if2d=if2d)

    def forward(self, x1):
        # 上采样
        x2 = self.up(x1)
        # 经历卷积
        x = self.conv(x2)
        return x


# 下采样
def down(in_channels, out_channels, if2d=True):
    # 池化 + 双卷积
    layer = []
    layer.append(conv(in_channels, out_channels, if2d=if2d))
    if if2d:
        layer.append(nn.MaxPool2d(2, stride=2))
    else:
        layer.append(nn.MaxPool1d(2, stride=2))
    return nn.Sequential(*layer)


class Residual_1D(nn.Module):
    def __init__(self, chnl_in, chnl_out):
        super(Residual_1D, self).__init__()

        self.conv1 = conv(chnl_in, chnl_out//2, kernlsize=1, pad=0, if2d=False)
        self.conv2 = conv(chnl_out//2, chnl_out // 2, if2d=False)
        self.conv3 = conv(chnl_out//2, chnl_out, kernlsize=1, pad=0, if2d=False)

        self.conv_side = nn.Conv1d(chnl_in, chnl_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x_side = self.conv_side(x)

        return x3+x_side


class Residual_2D(nn.Module):
    def __init__(self, chnl_in, chnl_out):
        super(Residual_2D, self).__init__()

        self.conv1 = conv(chnl_in, chnl_out // 2, kernlsize=1, pad=0, if2d=True)
        self.conv2 = conv(chnl_out // 2, chnl_out // 2, if2d=True)
        self.conv3 = conv(chnl_out // 2, chnl_out, kernlsize=1, pad=0, if2d=True)

        self.conv_side = nn.Conv2d(chnl_in, chnl_out, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x_side = self.conv_side(x)

        return x3 + x_side


class Resblock_1D(nn.Module):
    def __init__(self, chnl_in, chnl_out):
        super(Resblock_1D, self).__init__()

        self.res1 = Residual_1D(chnl_in, 256)
        self.res2 = Residual_1D(256, 256)
        self.res3 = Residual_1D(256, chnl_out)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)

        return x3


class Resblock_2D(nn.Module):
    def __init__(self, chnl_in, chnl_out):
        super(Resblock_2D, self).__init__()

        self.res1 = Residual_2D(chnl_in, 256)
        self.res2 = Residual_2D(256, 256)
        self.res3 = Residual_2D(256, chnl_out)

    def forward(self, x):
        x1 = self.res1(x)
        x2 = self.res2(x1)
        x3 = self.res3(x2)

        return x3


class Hourglass_1D(nn.Module):
    def __init__(self, io_chnl=15):
        super(Hourglass_1D, self).__init__()

        self.down1 = nn.MaxPool1d(2, stride=2)
        self.resblock1 = Resblock_1D(io_chnl, 256)
        self.down2 = nn.MaxPool1d(2, stride=2)
        self.resblock2 = Resblock_1D(256, 256)
        self.down3 = nn.MaxPool1d(2, stride=2)
        self.resblock3 = Resblock_1D(256, 256)

        self.side1 = Resblock_1D(io_chnl, io_chnl)
        self.side2 = Resblock_1D(256, io_chnl)  # 256@32 -> 16@32
        self.side3 = Resblock_1D(256, io_chnl)  # 256@32 -> 16@32

        self.res_out = Residual_1D(256, io_chnl)
        self.res_out_3 = Residual_1D(io_chnl, io_chnl)
        self.up3 = Up(io_chnl, io_chnl, if2d=False)  # 16@16 -> 16@32
        self.res_out_2 = Residual_1D(io_chnl, io_chnl)
        self.up2 = Up(io_chnl, io_chnl, if2d=False)  # 16@16 -> 16@32
        self.res_out_1 = Residual_1D(io_chnl, io_chnl)
        self.up1 = Up(io_chnl, io_chnl, if2d=False)  # 16@32 -> 16@64

    def forward(self, x):
        x1_1 = self.down1(x)
        x1_2 = self.resblock1(x1_1)
        x2_1 = self.down2(x1_2)
        x2_2 = self.resblock2(x2_1)
        x3_1 = self.down3(x2_2)
        x3_2 = self.resblock2(x3_1)

        x1_s = self.side1(x)
        x2_s = self.side2(x1_2)
        x3_s = self.side3(x2_2)

        x0_out = self.res_out(x3_2)
        x3_out1 = self.res_out_3(x0_out)
        x3_out2 = self.up3(x3_out1)
        x2_out1 = self.res_out_2(x3_out2 + x3_s)
        x2_out2 = self.up2(x2_out1)
        x1_out1 = self.res_out_1(x2_out2 + x2_s)
        x1_out2 = self.up1(x1_out1)

        return x1_s + x1_out2


class Hourglass_2D(nn.Module):
    def __init__(self, io_chnl=15):
        super(Hourglass_2D, self).__init__()
        self.down1 = nn.MaxPool2d(2, stride=2)  # 15@8*8 => 15@4*4
        self.resblock1 = Resblock_2D(io_chnl, 256)
        self.down2 = nn.MaxPool2d(2, stride=2)  # 256@4*4 -> 256@2*2
        self.resblock2 = Resblock_2D(256, 256)
        self.down3 = nn.MaxPool2d(2, stride=2)  # 256@2*2 -> 256@1*1
        self.resblock3 = Resblock_2D(256, 256)

        self.side1 = Resblock_2D(io_chnl, io_chnl)
        self.side2 = Resblock_2D(256, io_chnl)  # 256@32 -> 16@32
        self.side3 = Resblock_2D(256, io_chnl)  # 256@32 -> 16@32

        self.res_out = Residual_2D(256, io_chnl)
        self.res_out_3 = Residual_2D(io_chnl, io_chnl)
        self.up3 = Up(io_chnl, io_chnl, if2d=True)  # 16@16 -> 16@32
        self.res_out_2 = Residual_2D(io_chnl, io_chnl)
        self.up2 = Up(io_chnl, io_chnl, if2d=True)  # 16@16 -> 16@32
        self.res_out_1 = Residual_2D(io_chnl, io_chnl)
        self.up1 = Up(io_chnl, io_chnl, if2d=True)  # 16@32 -> 16@64

    def forward(self, x):
        x1_1 = self.down1(x)  # 15@32*32 -> 15@16*16
        x1_2 = self.resblock1(x1_1)
        x2_1 = self.down2(x1_2)  # 256@16*16 -> 256@8*8
        x2_2 = self.resblock2(x2_1)
        x3_1 = self.down3(x2_2)  # 256@16*16 -> 256@8*8
        x3_2 = self.resblock2(x3_1)

        x1_s = self.side1(x)
        x2_s = self.side2(x1_2)
        x3_s = self.side3(x2_2)  # 16@64 -> 16@64

        x0_out = self.res_out(x3_2)
        x3_out1 = self.res_out_3(x0_out)
        x3_out2 = self.up3(x3_out1)
        x2_out1 = self.res_out_2(x3_out2 + x3_s)
        x2_out2 = self.up2(x2_out1)
        x1_out1 = self.res_out_1(x2_out2 + x2_s)
        x1_out2 = self.up1(x1_out1)

        return x1_s + x1_out2


class PoseNet(nn.Module):
    def __init__(self, stg, flag_train=0, io_channels=160, base_channel=30, gru_channel_out=15):
        """
        :param in_size: 输入大小，()
        :param out_channels: 卷积后输出的通道数
        :param if2d: Whether the dimension of convolution is 2d, bool
        :return:
        """
        super(PoseNet, self).__init__()
        self.stg = stg
        self.flag_train = flag_train
        self.basechannel = gru_channel_out
        self.base_channel = base_channel
        self.io_channels = io_channels

        if self.stg == 0:
            # 下采样
            self.down1 = down(self.io_channels, self.io_channels, if2d=False)  # 160@96 => 160@48 -> 30@16*16
            self.up1 = Up(self.io_channels, self.io_channels, if2d=False)  # 160@48 => 160@96

        elif self.stg == 1:
            self.down2 = down(self.base_channel, self.base_channel)  # 30@16*16 => 30@8*8
            # 上采样
            self.up2 = Up(self.base_channel, self.base_channel)  # out: 30@8*8 => 30@16*16 -> 160@48

        elif self.stg == 2:
            # ConvGRU
            self.convGRU1 = ConvGRUCell(input_dim=self.base_channel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 30@8*8=> 15@8*8
            self.convGRU2 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 15@8*8
            self.convGRU3 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 15@8*8
            self.convGRU4 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.base_channel,
                                        kernel_size=(3, 3))  # 15@8*8=> 30@8*8

        elif self.stg == 3:
            # 下采样
            self.down1 = down(self.io_channels, self.io_channels, if2d=False)  # 160@96 => 160@48 -> 30@16*16
            self.down2 = down(self.base_channel, self.base_channel)  # 30@16*16 => 30@8*8
            # ConvGRU
            self.convGRU1 = ConvGRUCell(input_dim=self.base_channel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 30@8*8=> 15@8*8
            self.convGRU2 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 15@8*8
            self.convGRU3 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 15@8*8
            self.convGRU4 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.base_channel,
                                        kernel_size=(3, 3))  # 15@8*8=> 30@8*8
            # 上采样
            self.up2 = Up(self.base_channel, self.base_channel)  # out: 30@8*8 => 30@16*16 -> 160@48
            self.up1 = Up(self.io_channels, self.io_channels, if2d=False)  # 160@48 => 160@96

        elif self.stg == 4:
            # 下采样
            self.down1 = down(self.io_channels, self.io_channels, if2d=False)  # 160@96 => 160@48 -> 30@16*16
            self.down2 = down(self.base_channel, self.base_channel)  # 30@16*16 => 30@8*8
            # ConvGRU
            self.convGRU1 = ConvGRUCell(input_dim=self.base_channel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 30@8*8=> 15@8*8
            self.convGRU2 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 15@8*8

            # hourglass
            self.hourglass2d = Hourglass_2D()  # 15@8*8

            # depth
            self.hourglass1d_1 = Hourglass_1D()  # 15@8*8 -> 15@64 => 15@64
            self.conv1_1 = conv(self.basechannel, self.basechannel, kernlsize=3, pad=1, if2d=False)  # 15@64
            self.conv1_2 = conv(self.basechannel, self.basechannel, kernlsize=3, pad=1, if2d=False)  # 15@64
            self.depth_out = nn.Sigmoid()  # 15@64

            # 2D HM
            self.up_1 = Up(self.basechannel, self.basechannel)  # 15@8*8 => 15@16*16
            self.up_2 = Up(self.basechannel, self.basechannel)  # 15@16*16 => 15@32*32
            self.up_3 = Up(self.basechannel, self.basechannel)  # 15@32*32 => 15@64*64
            self.HM_out = nn.Sigmoid()  # 15@64*64

            # 2D Position
            self.up_4 = Up(self.basechannel, self.basechannel)  # 15@8*8 => 15@16*16
            self.up_5 = Up(self.basechannel, self.basechannel)  # 15@16*16 => 15@32*32
            self.up_6 = Up(self.basechannel, self.basechannel)  # 15@32*32 => 15@64*64
            self.up_7 = Up(self.basechannel, 1)  # 15@64*64 => 1@128*128
            self.P_out = nn.Sigmoid()  # 15@128*128


        elif self.stg == 5:
            # 下采样
            self.down1 = down(self.io_channels, self.io_channels, if2d=False)  # 160@96 => 160@48 -> 30@16*16
            self.down2 = down(self.base_channel, self.base_channel)  # 30@16*16 => 30@8*8
            # ConvGRU
            self.convGRU1 = ConvGRUCell(input_dim=self.base_channel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 30@8*8=> 15@8*8
            self.convGRU2 = ConvGRUCell(input_dim=self.basechannel, output_dim=self.basechannel,
                                        kernel_size=(3, 3))  # 15@8*8

            # hourglass
            self.hourglass2d = Hourglass_2D()  # 15@8*8

            # 2D Position
            self.up_4 = Up(self.basechannel, self.basechannel)  # 15@8*8 => 15@16*16
            self.up_5 = Up(self.basechannel, self.basechannel)  # 15@16*16 => 15@32*32
            self.up_6 = Up(self.basechannel, self.basechannel)  # 15@32*32 => 15@64*64
            self.up_7 = Up(self.basechannel, 1)  # 15@64*64 => 1@128*128
            self.P_out = nn.Sigmoid()  # 15@128*128

        elif self.stg == 6:
            self.conv1_out = conv(self.base_channel, self.basechannel, kernlsize=3, pad=1, if2d=True)  # 15@64
            self.conv1_out_2 = conv(self.basechannel, self.base_channel, kernlsize=3, pad=1, if2d=True)  # 15@64

        elif self.stg == 7:
            # 下采样
            self.down1 = down(self.io_channels, self.io_channels, if2d=False)  # 160@96 => 160@48 -> 30@16*16
            self.down2 = down(self.base_channel, self.base_channel)  # 30@16*16 => 30@8*8
            # Conv
            self.conv1_out = conv(self.base_channel, self.basechannel, kernlsize=3, pad=1, if2d=True)  # 15@64
            self.conv1_out_2 = conv(self.basechannel, self.base_channel, kernlsize=3, pad=1, if2d=True)  # 15@64
            # 上采样
            self.up2 = Up(self.base_channel, self.base_channel)  # out: 30@8*8 => 30@16*16 -> 160@48
            self.up1 = Up(self.io_channels, self.io_channels, if2d=False)  # 160@48 => 160@96

        else:
            # 下采样
            self.down1 = down(self.io_channels, self.io_channels, if2d=False)  # 160@96 => 160@48 -> 30@16*16
            self.down2 = down(self.base_channel, self.base_channel)  # 30@16*16 => 30@8*8
            # Conv
            self.conv1_out = conv(self.base_channel, self.basechannel, kernlsize=3, pad=1, if2d=True)  # 15@64

            # hourglass
            self.hourglass2d = Hourglass_2D()  # 15@8*8

            # depth
            self.hourglass1d_1 = Hourglass_1D()  # 15@8*8 -> 15@64 => 15@64
            self.conv1_1 = conv(self.basechannel, self.basechannel, kernlsize=3, pad=1, if2d=False)  # 15@64
            self.conv1_2 = conv(self.basechannel, self.basechannel, kernlsize=3, pad=1, if2d=False)  # 15@64
            self.depth_out = nn.Sigmoid()  # 15@64

            # 2D HM
            self.up_1 = Up(self.basechannel, self.basechannel)  # 15@8*8 => 15@16*16
            self.up_2 = Up(self.basechannel, self.basechannel)  # 15@16*16 => 15@32*32
            self.up_3 = Up(self.basechannel, self.basechannel)  # 15@32*32 => 15@64*64
            self.HM_out = nn.Sigmoid()  # 15@64*64

            # 2D Position
            self.up_4 = Up(self.basechannel, self.basechannel)  # 15@8*8 => 15@16*16
            self.up_5 = Up(self.basechannel, self.basechannel)  # 15@16*16 => 15@32*32
            self.up_6 = Up(self.basechannel, self.basechannel)  # 15@32*32 => 15@64*64
            self.up_7 = Up(self.basechannel, 1)  # 15@64*64 => 1@128*128
            self.P_out = nn.Sigmoid()  # 15@128*128

    def forward(self, x, h=None):
        if h is None:
            h = [None, None, None, None]

        if self.stg == 0:
            if self.flag_train == 1:
                x_e1 = self.down1(x)  # 160@96 => 160@48
                x_e1_2 = ((x_e1.reshape(-1, 10, 256, 3)).permute(0, 1, 3, 2)).reshape(-1, 30, 16,
                                                                                      16)  # 160@48 -> 30@16*16
                x_d2_2 = ((x_e1_2.reshape(-1, 10, 3, 256)).permute(0, 1, 3, 2)).reshape(-1, 160,
                                                                                        48)  # 30@16*16 -> 160@48
                out = self.up1(x_d2_2)  # 160@48 => 160@96
            else:
                x_e1 = self.down1(x)  # 160@96 => 160@48
                out = ((x_e1.reshape(-1, 10, 256, 3)).permute(0, 1, 3, 2)).reshape(-1, 30, 16,
                                                                                      16)  # 160@48 -> 30@16*16
            return {'out': out}

        elif self.stg == 1:
            if self.flag_train == 1:
                x_e2 = self.down2(x)  # 30@16*16 => 30@8*8
                out = self.up2(x_e2)  # 30@8*8 => 30@16*16
            else:
                out = self.down2(x)  # 30@16*16 => 30@8*8
            return {'out': out}

        elif self.stg == 2:
            if self.flag_train == 1:
                x_e3 = self.convGRU1(x, h[0])
                x_e4 = self.convGRU2(x_e3, h[1])
                x_d4 = self.convGRU3(x_e4, h[2])
                out = self.convGRU4(x_d4, h[3])
            else:
                x_e3 = self.convGRU1(x, h[0])
                out = self.convGRU2(x_e3, h[1])
            return {'out': out}

        elif self.stg == 3:
            # Encoder
            x_e1 = self.down1(x)  # 160@96 => 160@48
            x_e1_2 = ((x_e1.reshape(-1, 10, 256, 3)).permute(0, 1, 3, 2)).reshape(-1, 30, 16,
                                                                                  16)  # 160@48 -> 30@16*16
            x_e2 = self.down2(x_e1_2)  # 30@16*16 => 30@8*8
            x_e3 = self.convGRU1(x_e2, h[0])
            x_e4 = self.convGRU2(x_e3, h[1])

            # Decoder
            x_d4 = self.convGRU3(x_e4, h[2])
            x_d3 = self.convGRU4(x_d4, h[3])
            x_d2 = self.up2(x_d3)  # 30@8*8 => 30@16*16
            x_d2_2 = ((x_d2.reshape(-1, 10, 3, 256)).permute(0, 1, 3, 2)).reshape(-1, 160,
                                                                                    48)  # 30@16*16 -> 160@48
            out = self.up1(x_d2_2)  # 160@48 => 160@96
            return {'out': out}

        elif self.stg == 4:
            # Encoder
            x_e1 = self.down1(x)  # 160@96 => 160@48
            x_e1_2 = ((x_e1.reshape(-1, 10, 256, 3)).permute(0, 1, 3, 2)).reshape(-1, 30, 16,
                                                                                  16)  # 160@48 -> 30@16*16
            x_e2 = self.down2(x_e1_2)  # 30@16*16 => 30@8*8
            x_e3 = self.convGRU1(x_e2, h[0])
            x_e4 = self.convGRU2(x_e3, h[1])

            x_hg = self.hourglass2d(x_e4)

            # depth
            x_d1 = x_hg.reshape(-1, 15, 64)
            x_d2 = self.hourglass1d_1(x_d1)
            x_d3 = self.conv1_1(x_d2)
            x_d4 = self.conv1_2(x_d3)
            x_d5 = self.depth_out(x_d4)

            # 2D HeatMap
            x_h1 = self.up_1(x_hg)  # 15@8*8 => 15@16*16
            x_h2 = self.up_2(x_h1)  # 15@16*16 => 15@32*32
            x_h3 = self.up_3(x_h2)  # 15@32*32 => 15@64*64
            x_h4 = self.HM_out(x_h3)

            # 2D Position
            x_p1 = self.up_4(x_hg)  # 15@8*8 => 15@16*16
            x_p2 = self.up_5(x_p1)  # 15@16*16 => 15@32*32
            x_p3 = self.up_6(x_p2)  # 15@32*32 => 15@64*64
            x_p4 = self.up_7(x_p3)  # 15@64*64 => 1@128*128
            x_p5 = self.P_out(x_p4)

            return {'Depth': x_d5,
                    'HeatMap': x_h4,
                    'Position': x_p5}

        elif self.stg == 5:
            # Encoder
            x_e1 = self.down1(x)  # 160@96 => 160@48
            x_e1_2 = ((x_e1.reshape(-1, 10, 256, 3)).permute(0, 1, 3, 2)).reshape(-1, 30, 16,
                                                                                  16)  # 160@48 -> 30@16*16
            x_e2 = self.down2(x_e1_2)  # 30@16*16 => 30@8*8
            x_e3 = self.convGRU1(x_e2, h[0])
            x_e4 = self.convGRU2(x_e3, h[1])

            x_hg = self.hourglass2d(x_e4)

            # 2D Position
            x_p1 = self.up_4(x_hg)  # 15@8*8 => 15@16*16
            x_p2 = self.up_5(x_p1)  # 15@16*16 => 15@32*32
            x_p3 = self.up_6(x_p2)  # 15@32*32 => 15@64*64
            x_p4 = self.up_7(x_p3)  # 15@64*64 => 1@128*128
            x_p5 = self.P_out(x_p4)

            return {'Position': x_p5}

        elif self.stg == 6:
            x_e3 = self.conv1_out(x)
            out = self.conv1_out_2(x_e3)

            return {'out': out}

        elif self.stg == 7:
            # Encoder
            x_e1 = self.down1(x)  # 160@96 => 160@48
            x_e1_2 = ((x_e1.reshape(-1, 10, 256, 3)).permute(0, 1, 3, 2)).reshape(-1, 30, 16,
                                                                                  16)  # 160@48 -> 30@16*16
            x_e2 = self.down2(x_e1_2)  # 30@16*16 => 30@8*8
            x_e3 = self.conv1_out(x_e2)

            # Decoder
            x_d3 = self.conv1_out_2(x_e3)
            x_d2 = self.up2(x_d3)  # 30@8*8 => 30@16*16
            x_d2_2 = ((x_d2.reshape(-1, 10, 3, 256)).permute(0, 1, 3, 2)).reshape(-1, 160,
                                                                                  48)  # 30@16*16 -> 160@48
            out = self.up1(x_d2_2)  # 160@48 => 160@96
            return {'out': out}

        else:
            # Encoder
            x_e1 = self.down1(x)  # 160@96 => 160@48
            x_e1_2 = ((x_e1.reshape(-1, 10, 256, 3)).permute(0, 1, 3, 2)).reshape(-1, 30, 16,
                                                                                  16)  # 160@48 -> 30@16*16
            x_e2 = self.down2(x_e1_2)  # 30@16*16 => 30@8*8
            x_e3 = self.conv1_out(x_e2)

            x_hg = self.hourglass2d(x_e3)

            # depth
            x_d1 = x_hg.reshape(-1, 15, 64)
            x_d2 = self.hourglass1d_1(x_d1)
            x_d3 = self.conv1_1(x_d2)
            x_d4 = self.conv1_2(x_d3)
            x_d5 = self.depth_out(x_d4)

            # 2D HeatMap
            x_h1 = self.up_1(x_hg)  # 15@8*8 => 15@16*16
            x_h2 = self.up_2(x_h1)  # 15@16*16 => 15@32*32
            x_h3 = self.up_3(x_h2)  # 15@32*32 => 15@64*64
            x_h4 = self.HM_out(x_h3)

            # 2D Position
            x_p1 = self.up_4(x_hg)  # 15@8*8 => 15@16*16
            x_p2 = self.up_5(x_p1)  # 15@16*16 => 15@32*32
            x_p3 = self.up_6(x_p2)  # 15@32*32 => 15@64*64
            x_p4 = self.up_7(x_p3)  # 15@64*64 => 1@128*128
            x_p5 = self.P_out(x_p4)

            return {'Depth': x_d5,
                    'HeatMap': x_h4,
                    'Position': x_p5}
