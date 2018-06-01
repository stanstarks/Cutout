'''Fixed ENAS cell in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def sep_bn(channels, kernel_size, stride=1):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=False, groups=channels),
        nn.BatchNorm2d(channels),
        nn.ReLU6())


def conv_bn(in_c, out_c, kernel_size, stride=1):
    if kernel_size == 3:
        padding = 1
    else:
        padding = 0
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU6())


class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)


class EnasCell(nn.Module):
    def __init__(self, x_branch, y_branch):
        super(EnasCell, self).__init__()
        self.x_op, self.x_id = x_branch
        self.y_op, self.y_id = y_branch

    def forward(self, prev_layers):
        def _output(op, prev_id, prev_layers):
            if op == 'Empty':
                return 0
            if op == 'Identity':
                return prev_layers[prev_id]
            return op(prev_layers[prev_id])

        x_out = _output(self.x_op, self.x_id, prev_layers)
        y_out = _output(self.y_op, self.y_id, prev_layers)
        return x_out + y_out


class EnasBlock(nn.Module):
    conv_ops = [0, 1, 2]

    def __init__(self, in_c, out_c, arc, num_cells=5):
        """ Create ENAS block.
        """
        super(EnasBlock, self).__init__()
        self.num_cells = num_cells
        self.used = [0] * (num_cells + 2)
        cells = []
        for cell_id in range(num_cells):
            x_id = arc[4 * cell_id]
            self.used[x_id] = 1
            x_op = arc[4 * cell_id + 1]
            x_branch = (self._make_branch(x_op, in_c, out_c, x_id), x_id)
            y_id = arc[4 * cell_id + 2]
            self.used[y_id] = 1
            y_op = arc[4 * cell_id + 3]
            y_branch = (self._make_branch(y_op, in_c, out_c, y_id), y_id)
            cells.append(EnasCell(x_branch, y_branch))
        self.cells = ListModule(*cells)
        self.out_channels = (num_cells + 2 - sum(self.used)) * out_c

    def _make_branch(self, op, in_c, out_c, prev_id, inverse=False):
        def _make_op(op, channels):
            if op < 2:
                return sep_bn(channels, 3)
            if op == 2:
                return sep_bn(channels, 5)
            if op == 3:
                return nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
            if op == 4:
                return nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
            if op == 5:
                return 'Identity'

        """create one cell branch.
        inverse: if True, transform after op."""

        if op == 6:
            return 'Empty'
        if prev_id > 1 or in_c == out_c:
            return _make_op(op, in_c)
        transform = conv_bn(in_c, out_c, 1)
        if op == 5:
            return transform
        if inverse and op in self.conv_ops:
            comp = _make_op(op, in_c)
            return nn.Sequential(comp, transform)
        else:
            comp = _make_op(op, out_c)
            return nn.Sequential(transform, comp)

    def forward(self, layers):
        # Currently don't support dropping branch

        prev_layers = layers
        for cell in self.cells:
            prev_layers.append(cell(prev_layers))

        outs = [prev_layers[k] for k, v in enumerate(self.used) if v == 0]
        return torch.cat(outs, dim=1)


class Enas(nn.Module):
    def __init__(self, arcs, out_filters=20, num_layers=6, num_classes=10):
        super(Enas, self).__init__()
        self.num_layers = num_layers
        pool_num = 3
        pool_distance = self.num_layers // pool_num
        self.pool_layers = [i * pool_distance + 2 for i in range(pool_num)]
        enas_blocks = []
        enas_outputs = []
        reduction_blocks = []

        self.stem_conv = conv_bn(3, out_filters, 3)
        for layer_id in range(self.num_layers + 2):
            if layer_id in self.pool_layers:
                # add reduction
                reduction_blocks.append(conv_bn(out_filters // 2, out_filters,
                                                2, stride=2))
                arc = arcs[1]
            else:
                arc = arcs[0]
            enas_block = EnasBlock(out_filters, out_filters, arc)
            enas_blocks.append(enas_block)
            if layer_id + 1 in self.pool_layers:
                # upsample in advance
                out_filters *= 2
                kernel_size, stride = 2, 2
            else:
                kernel_size, stride = 1, 1
            enas_outputs.append(conv_bn(enas_block.out_channels, out_filters,
                                        kernel_size, stride=stride))
        # output no auxiliary
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(enas_block.out_channels, num_classes)
        self.enas_blocks = ListModule(*enas_blocks)
        self.enas_outputs = ListModule(*enas_outputs)
        self.reduction_blocks = ListModule(*reduction_blocks)


    def forward(self, x):
        x = self.stem_conv(x)
        layers = [x, x]
        red_id = 0
        for layer_id in range(self.num_layers + 2):
            if layer_id in self.pool_layers:
                layers[0] = self.reduction_blocks[red_id](layers[0])
                red_id += 1
            x = self.enas_blocks[layer_id](layers)
            layers = [layers[1], self.enas_outputs[layer_id](x)]
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out
