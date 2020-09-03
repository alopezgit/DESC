import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torch.autograd import Function
from utils.bilinear_sampler import *

##############################################################################
# Initializers and helpers
##############################################################################


def get_nonlinearity_layer(activation_type='PReLU'):
    if activation_type == 'ReLU':
        nonlinearity_layer = nn.ReLU(True)
    elif activation_type == 'SELU':
        nonlinearity_layer = nn.SELU(True)
    elif activation_type == 'LeakyReLU':
        nonlinearity_layer = nn.LeakyReLU(0.1, True)
    elif activation_type == 'PReLU':
        nonlinearity_layer = nn.PReLU()
    else:
        raise NotImplementedError(
            'activation layer [%s] is not found' % activation_type)
    return nonlinearity_layer


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net



##############################################################################
# Losses
##############################################################################


def gradient_x(img):
    gx = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gx


def gradient_y(img):
    gy = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gy


def ssim(x, y):

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(x, 3, 1)
    mu_y = F.avg_pool2d(y, 3, 1)

    sigma_x = F.avg_pool2d(x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y, 3, 1) - mu_x*mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d
    return torch.clamp((1-SSIM)/2, 0, 1)


class SmoothLoss(nn.Module):
    # Smoothing loss which smooths the predictions based on the
    # RGB gradient.
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, depth, image, sem_seg=0):
        depth_grad_x = gradient_x(depth)
        depth_grad_y = gradient_y(depth)
        image_grad_x = gradient_x(image)
        image_grad_y = gradient_y(image)
        if sem_seg == 1:
            image_grad_x = (image_grad_x > 0) * 10.0
            image_grad_y = (image_grad_y > 0) * 10.0
        if sem_seg == 2:
            image_grad_x = (image_grad_x == 0) * 100.0
            image_grad_y = (image_grad_y == 0) * 100.0
        weights_x = torch.exp(-torch.mean(torch.abs(image_grad_x), 1, True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_grad_y), 1, True))
        smoothness_x = depth_grad_x*weights_x
        smoothness_y = depth_grad_y*weights_y
        if sem_seg == 2:
            smoothness_x = -smoothness_x
            smoothness_y = -smoothness_y
        loss_x = torch.mean(torch.abs(smoothness_x))
        loss_y = torch.mean(torch.abs(smoothness_y))

        loss = loss_x + loss_y

        return loss


class ReconLoss(nn.Module):
    # Photometric loss used for the stereo supervision case
    # This loss is used in GASDA
    def __init__(self, alpha=0.85):
        super(ReconLoss, self).__init__()
        self.alpha = alpha

    def forward(self, img0, img1, pred, fb, max_d=655.35):

        x0 = (img0 + 1.0) / 2.0
        x1 = (img1 + 1.0) / 2.0

        assert x0.shape[0] == pred.shape[0]
        assert pred.shape[0] == fb.shape[0]

        new_depth = (pred + 1.0) / 2.0
        new_depth *= max_d
        disp = 1.0 / (new_depth+1e-6)
        tmp = np.array(fb)
        for i in range(new_depth.shape[0]):
            disp[i, :, :, :] *= tmp[i]
            disp[i, :, :, :] /= disp.shape[3]  # normalize to [0,1]

        x0_w = bilinear_sampler_1d_h(x1, -1.0*disp)

        ssim_ = ssim(x0, x0_w)
        l1 = torch.abs(x0-x0_w)
        loss1 = torch.mean(self.alpha * ssim_)
        loss2 = torch.mean((1-self.alpha) * l1)
        loss = loss1 + loss2

        recon_img = x0_w * 2.0-1.0

        return loss, recon_img


##############################################################################
# Networks
##############################################################################

# Below there are the networks and their building blocks used for our models
# These blocks are taken from GASDA/T2Net


class GaussianNoiseLayer(nn.Module):
    def __init__(self):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable((torch.randn(x.size()).cuda(
            x.data.get_device()) - 0.5) / 10.0)
        return x+noise


class _InceptionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), width=1, drop_rate=0, use_bias=False):
        super(_InceptionBlock, self).__init__()

        self.width = width
        self.drop_rate = drop_rate

        for i in range(width):
            layer = nn.Sequential(
                nn.ReflectionPad2d(i*2+1),
                nn.Conv2d(input_nc, output_nc, kernel_size=3,
                          padding=0, dilation=i*2+1, bias=use_bias)
            )
            setattr(self, 'layer'+str(i), layer)

        self.norm1 = norm_layer(output_nc * width)
        self.norm2 = norm_layer(output_nc)
        self.nonlinearity = nonlinearity
        self.branch1x1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(output_nc * width, output_nc,
                      kernel_size=3, padding=0, bias=use_bias)
        )

    def forward(self, x):
        result = []
        for i in range(self.width):
            layer = getattr(self, 'layer'+str(i))
            result.append(layer(x))
        output = torch.cat(result, 1)
        output = self.nonlinearity(self.norm1(output))
        output = self.norm2(self.branch1x1(output))
        if self.drop_rate > 0:
            output = F.dropout(output, p=self.drop_rate,
                               training=self.training)

        return self.nonlinearity(output+x)


class _EncoderBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_EncoderBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, middle_nc, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.Conv2d(middle_nc, output_nc, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _DownBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DownBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, output_nc, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity,
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _ShuffleUpBlock(nn.Module):
    def __init__(self, input_nc, up_scale, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_ShuffleUpBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, input_nc*up_scale**2, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            nn.PixelShuffle(up_scale),
            nonlinearity,
            nn.Conv2d(input_nc, output_nc, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _DecoderUpBlock(nn.Module):
    def __init__(self, input_nc, middle_nc, output_nc, norm_layer=nn.BatchNorm2d, nonlinearity=nn.PReLU(), use_bias=False):
        super(_DecoderUpBlock, self).__init__()

        model = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_nc, middle_nc, kernel_size=3,
                      stride=1, padding=0, bias=use_bias),
            norm_layer(middle_nc),
            nonlinearity,
            nn.ConvTranspose2d(middle_nc, output_nc, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            norm_layer(output_nc),
            nonlinearity
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class _OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=3, use_bias=False):
        super(_OutputBlock, self).__init__()
        model = [
            nn.ReflectionPad2d(int(kernel_size/2)),
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size,
                      padding=0, bias=use_bias),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ngf=64, layers=4, norm='batch', drop_rate=0, add_noise=False, weight=0.1):
        super(UNetGenerator, self).__init__()

        self.layers = layers
        self.weight = weight
        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type='PReLU')

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # encoder part
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        )
        self.conv2 = _EncoderBlock(
            ngf, ngf*2, ngf*2, norm_layer, nonlinearity, use_bias)
        self.conv3 = _EncoderBlock(
            ngf*2, ngf*4, ngf*4, norm_layer, nonlinearity, use_bias)
        self.conv4 = _EncoderBlock(
            ngf*4, ngf*8, ngf*8, norm_layer, nonlinearity, use_bias)

        for i in range(layers-4):
            conv = _EncoderBlock(ngf*8, ngf*8, ngf*8,
                                 norm_layer, nonlinearity, use_bias)
            setattr(self, 'down'+str(i), conv.model)

        center = []
        for i in range(7-layers):
            center += [
                _InceptionBlock(ngf*8, ngf*8, norm_layer,
                                nonlinearity, 7-layers, drop_rate, use_bias)
            ]

        center += [
            _DecoderUpBlock(ngf*8, ngf*8, ngf*4, norm_layer,
                            nonlinearity, use_bias)
        ]
        if add_noise:
            center += [GaussianNoiseLayer()]
        self.center = nn.Sequential(*center)

        for i in range(layers-4):
            upconv = _DecoderUpBlock(
                ngf*(8+4), ngf*8, ngf*4, norm_layer, nonlinearity, use_bias)
            setattr(self, 'up' + str(i), upconv.model)

        self.deconv4 = _DecoderUpBlock(
            ngf*(4+4), ngf*8, ngf*2, norm_layer, nonlinearity, use_bias)
        self.deconv3 = _DecoderUpBlock(
            ngf*(2+2)+output_nc, ngf*4, ngf, norm_layer, nonlinearity, use_bias)
        self.deconv2 = _DecoderUpBlock(
            ngf*(1+1)+output_nc, ngf*2, int(ngf/2), norm_layer, nonlinearity, use_bias)

        self.output4 = _OutputBlock(ngf*(4+4), output_nc, 3, use_bias)
        self.output3 = _OutputBlock(
            ngf*(2+2)+output_nc, output_nc, 3, use_bias)
        self.output2 = _OutputBlock(
            ngf*(1+1)+output_nc, output_nc, 3, use_bias)
        self.output1 = _OutputBlock(
            int(ngf/2)+output_nc, output_nc, 7, use_bias)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, input):
        conv1_s = self.conv1(input)
        conv1 = self.pool(conv1_s)
        conv2 = self.pool(self.conv2.forward(conv1))
        conv3 = self.pool(self.conv3.forward(conv2))
        center_in = self.pool(self.conv4.forward(conv3))

        middle = [center_in]
        for i in range(self.layers-4):
            model = getattr(self, 'down'+str(i))
            center_in = self.pool(model.forward(center_in))
            middle.append(center_in)
        center_out = self.center.forward(center_in)

        for i in range(self.layers-4):
            model = getattr(self, 'up'+str(i))
            center_out = model.forward(
                torch.cat([center_out, middle[self.layers-5-i]], 1))

        scale = 1.0
        result = []

        deconv4 = self.deconv4.forward(
            torch.cat([center_out, conv3 * self.weight], 1))
        output4 = scale * \
            self.output4.forward(
                torch.cat([center_out, conv3 * self.weight], 1))
        result.append(output4)
        deconv3 = self.deconv3.forward(
            torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        output3 = scale * self.output3.forward(
            torch.cat([deconv4, conv2 * self.weight * 0.5, self.upsample(output4)], 1))
        result.append(output3)
        deconv2 = self.deconv2.forward(
            torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        output2 = scale * self.output2.forward(
            torch.cat([deconv3, conv1 * self.weight * 0.1, self.upsample(output3)], 1))
        result.append(output2)
        output1 = scale * \
            self.output1.forward(
                torch.cat([deconv2, self.upsample(output2)], 1))
        result.append(output1)

        return result

# The code below includes the generator and discriminator used in T2Net


class _ResGeneratorT2Net(nn.Module):
    # ResNet generator used to translate rgb images from S->T in T2Net
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, norm='batch', activation='PReLU', drop_rate=0, add_noise=False, gpu_ids=[]):
        super(_ResGeneratorT2Net, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nonlinearity
        ]

        n_downsampling = 2
        mult = 1
        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (i+1), 2)
            encoder += [
                _EncoderBlock(ngf * mult_prev, ngf*mult, ngf*mult,
                              norm_layer, nonlinearity, use_bias),
                nn.AvgPool2d(kernel_size=2, stride=2)
            ]

        mult = min(2 ** n_downsampling, 2)
        for i in range(n_blocks-n_downsampling):
            encoder += [
                _InceptionBlock(ngf*mult, ngf*mult, norm_layer=norm_layer, nonlinearity=nonlinearity, width=1,
                                drop_rate=drop_rate, use_bias=use_bias)
            ]

        decoder = []
        if add_noise:
            decoder += [GaussianNoiseLayer()]

        for i in range(n_downsampling):
            mult_prev = mult
            mult = min(2 ** (n_downsampling - i - 1), 2)
            decoder += [
                _DecoderUpBlock(ngf*mult_prev, ngf*mult_prev,
                                ngf*mult, norm_layer, nonlinearity, use_bias),
            ]

        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, input):
        feature = self.encoder(input)
        result = [feature]
        output = self.decoder(feature)
        result.append(output)
        return result


class _MultiscaleDiscriminator(nn.Module):
    # ResNet discriminator used for adversarial loss in T2Net
    # We only use one scale (num_D=1), so we should merge this class
    # with the _Discriminator one below
    def __init__(self, input_nc, ndf=64, n_layers=3, num_D=1, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D
        self.gpu_ids = gpu_ids

        for i in range(num_D):
            netD = _Discriminator(input_nc, ndf, n_layers,
                                  norm, activation, gpu_ids)
            setattr(self, 'scale'+str(i), netD)

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[
                                       1, 1], count_include_pad=False)

    def forward(self, input):
        result = []
        for i in range(self.num_D):
            netD = getattr(self, 'scale'+str(i))
            output = netD.forward(input)
            result.append(output)
            if i != (self.num_D-1):
                input = self.downsample(input)
        return result


class _Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', activation='PReLU', gpu_ids=[]):
        super(_Discriminator, self).__init__()

        self.gpu_ids = gpu_ids

        norm_layer = get_norm_layer(norm_type=norm)
        nonlinearity = get_nonlinearity_layer(activation_type=activation)

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.Conv2d(input_nc, ndf, kernel_size=4,
                      stride=2, padding=1, bias=use_bias),
            nonlinearity,
        ]

        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            model += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=4,
                          stride=2, padding=1, bias=use_bias),
                norm_layer(ndf*nf_mult),
                nonlinearity,
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * 8),
            nonlinearity,
            nn.Conv2d(ndf*nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class HeightPredictor(nn.Module):
    # G_h network in the paper
    def __init__(self):
        super(HeightPredictor, self).__init__()

        self.net_prior_height_target = nn.Sequential(nn.Conv2d(1, 32, 3, stride=2), nn.ReLU(True),
                                                     nn.Conv2d(
                                                         32, 64, 3, stride=2), nn.ReLU(True),
                                                     nn.Conv2d(
                                                         64, 64, 3, stride=2), nn.ReLU(True),
                                                     nn.Conv2d(64, 64, 3, stride=2))
        self.net_prior_height_fc = nn.Sequential(nn.Linear(100, 64), nn.ReLU(
            True), nn.Linear(64, 64), nn.ReLU(True), nn.Linear(64, 1))
        self.net_prior_height_embed = nn.Embedding(80, 32)

    def forward(self, mask_inst, pred_class, height_box):
        feat_conv = self.net_prior_height_target(
            mask_inst).max(-1)[0].max(-1)[0]
        embed = self.net_prior_height_embed(pred_class)
        feat_fc = torch.cat([feat_conv, embed, height_box], 1)
        height_obj = self.net_prior_height_fc(feat_fc)
        return height_obj