import torch
from torch import nn as nn
from torch.nn import functional as F

from model.vgg_arch import VGGFeatureExtractor

_reduction_modes = ['none', 'mean', 'sum']


class GeneratorLoss_ESRGAN(nn.Module):
    def __init__(self):
        super(GeneratorLoss_ESRGAN, self).__init__()
        self.L1Loss = L1Loss()
        self.PerceptualLoss = PerceptualLoss()
        self.GANLoss = GANLoss()


    def forward(self, out_images, target_images, netD):

        l_g_total = 0

        # pixel loss
        l_g_pix = self.L1Loss(out_images, target_images)
        l_g_total += l_g_pix

        # perceptual loss
        l_g_percep, l_g_style = self.PerceptualLoss(out_images, target_images)
        if l_g_percep is not None:
            l_g_total += l_g_percep
        if l_g_style is not None:
            l_g_total += l_g_style

        # gan loss (relativistic gan)
        real_d_pred = netD(target_images).detach()
        fake_g_pred = netD(out_images)
        l_g_real = self.GANLoss(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
        l_g_fake = self.GANLoss(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
        l_g_gan = (l_g_real + l_g_fake) / 2
        l_g_total += l_g_gan

        return l_g_total


class DiscriminatorLoss_ESRGAN(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss_ESRGAN, self).__init__()
        self.GANLoss = GANLoss()

    def forward(self, out_images, target_images, netD):

        # optimize net_d
        # real
        real_d_pred = netD(target_images)
        l_d_real = self.GANLoss(real_d_pred, True, is_disc=True)

        # fake
        fake_d_pred = netD(out_images.detach().clone())  # clone for pt1.9
        l_d_fake = self.GANLoss(fake_d_pred, False, is_disc=True)

        return l_d_real, l_d_fake


def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='mean')


class L1Loss(nn.Module):

    def __init__(self, loss_weight=0.01):
        super(L1Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):

        return self.loss_weight * l1_loss(pred, target)


class PerceptualLoss(nn.Module):

    def __init__(self,
                 layer_weights={'conv5_4': 1},
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)
        self.criterion_type = criterion
        self.criterion = torch.nn.L1Loss()

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):

        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


class GANLoss(nn.Module):

    def __init__(self, gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0, loss_weight=0.005):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val
        self.loss = nn.BCEWithLogitsLoss()

    def _wgan_loss(self, input, target):

        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):

        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):

        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight



