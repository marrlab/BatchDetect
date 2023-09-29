from __future__ import print_function

from argparse import ArgumentParser
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

import histaugan.networks as networks
from histaugan.slice import bilateral_slice


class EfficientHistAuGAN(pl.LightningModule):
    def __init__(self, opts, learning_rate=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.opts = opts

        # manual optimization for more advanced training procedure
        self.automatic_optimization = False

        # initialize networks
        # - discriminator for adversarial training
        self.dis1 = networks.Discriminator(opts.input_dim, norm=opts.dis_norm, sn=opts.dis_spectral_norm,
                                           c_dim=opts.num_domains, image_size=opts.crop_size)
        self.dis2 = networks.Discriminator(opts.input_dim, norm=opts.dis_norm, sn=opts.dis_spectral_norm,
                                           c_dim=opts.num_domains, image_size=opts.crop_size)
        self.dis_content = networks.DiscriminatorContentLocalFeatures(c_dim=opts.num_domains)

        # - encoder
        params = {
            'luma_bins': 8,
            'channel_multiplier': 1,
            'spatial_bin': 16,
            'batch_norm': True,
            'net_input_size': self.opts.lowres,
            'guide_complexity': 16,
        }
        self.encoder = Encoder(params=params)

        # - generator
        self.generator = Generator(params=params)
        
        # initialize loss functions
        self.loss_contrastive = ContrastiveLoss(temperature=0.1, contrast_mode='all', contrastive_method='cl')
        
        # HistAuGAN for paired image training
        # self.histaugan = HistAuGAN.load_from_checkpoint('/lustre/groups/haicu/workspace/sophia.wagner/HistAuGAN-7sites-epoch=10-l1_cc_loss=0.86.ckpt')
        # del self.histaugan.dis1, self.histaugan.dis2, self.histaugan.dis_c
        
        # initialize network weights
        self.dis1.apply(networks.gaussian_weights_init)
        self.dis2.apply(networks.gaussian_weights_init)
        self.dis_content.apply(networks.gaussian_weights_init)

    def forward(self, x: torch.Tensor) -> Tuple[
            Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # notation as follows:
        # two paths, always with same content A and B
        # style a and b
        
        fullres, lowres, _ = x
        bs = fullres.shape[0]
        assert bs % 2 == 0, "Even batch size is required"

        half_size = bs // 2
        lowres_Aa = lowres[0:half_size]
        lowres_Bb = lowres[half_size:]
        fullres_Aa = fullres[0:half_size]
        fullres_Bb = fullres[half_size:]

        # get encodings
        real_img = torch.cat((lowres_Aa, lowres_Bb), 0)
        # z_content, z_attr = self.encoder.forward(real_img)
        z_content, (mu, log_var) = self.encoder.forward(real_img)

        # local content encoding
        z_content_A, z_content_B = torch.split(z_content, half_size, dim=0)

        # global style encoding
        std = log_var.mul(0.5).exp_()
        # eps = torch.randn(std.size(0), std.size(1)).to(image.device)
        eps = torch.randn_like(std)
        z_attr = eps.mul(std).add_(mu)
        z_attr_a, z_attr_b = torch.split(z_attr, half_size, dim=0)

        # get random z_a
        z_random = torch.randn_like(z_attr_a)

        # first cross translation and reconstruction
        input_content_forA = z_content_A.repeat((3, 1, 1, 1))
        input_content_forB = z_content_B.repeat((3, 1, 1, 1))
        input_attr_forA = torch.cat((z_attr_b, z_attr_a, z_random), 0)
        input_attr_forB = torch.cat((z_attr_a, z_attr_b, z_random), 0)
        
        output_fakeA = self.generator(fullres_Aa.repeat((3, 1, 1, 1)), input_content_forA, input_attr_forA)
        output_fakeB = self.generator(fullres_Bb.repeat((3, 1, 1, 1)), input_content_forB, input_attr_forB)
        fake_A_b, fake_A_a, fake_A_random = torch.split(output_fakeA, half_size, dim=0)
        fake_B_a, fake_B_b, fake_B_random = torch.split(output_fakeB, half_size, dim=0)

        # get reconstructed encodings
        fake_encoded_img = torch.cat((fake_A_b, fake_B_a), 0)
        fake_encoded_img_lowres = F.interpolate(fake_encoded_img, size=(self.opts.lowres, self.opts.lowres))
        # z_content_recon, z_attr_recon = self.encoder(fake_encoded_img_lowres)
        # - local content encoding
        z_content_recon, (mu_recon, log_var_recon) = self.encoder(fake_encoded_img_lowres)
        z_content_recon_A, z_content_recon_B = torch.split(z_content_recon, half_size, dim=0)
        # - global style encoding
        std_recon = log_var_recon.mul(0.5).exp_()
        # eps_recon = torch.randn(std_recon.size(0), std_recon.size(1)).to(image.device)
        eps_recon = torch.randn_like(std_recon)
        z_attr_recon = eps_recon.mul(std_recon).add_(mu_recon)
        z_attr_recon_b, z_attr_recon_a = torch.split(z_attr_recon, half_size, dim=0)

        # second cross translation
        fake_A_recon = self.generator(fake_A_b, z_content_recon_A, z_attr_recon_a)
        fake_B_recon = self.generator(fake_B_a, z_content_recon_B, z_attr_recon_b)

        # for latent regression with random attribute
        fake_random_img = torch.cat((fake_A_random, fake_B_random), 0)
        fake_random_img_lowres = F.interpolate(fake_random_img, size=(self.opts.lowres, self.opts.lowres))
        with torch.no_grad():
            # _, z_attr_random = self.encoder(fake_random_img_lowres)
            _, (mu_random, _) = self.encoder(fake_random_img_lowres)
        # z_attr_random_a, z_attr_random_b = torch.split(z_attr_random, half_size, 0)
        z_attr_random_a, z_attr_random_b = torch.split(mu_random, half_size, 0)

        return z_content, z_attr, mu, log_var, z_random, fake_encoded_img, fake_A_a, fake_B_b, fake_A_recon, \
            fake_B_recon, fake_random_img, z_attr_random_a, z_attr_random_b, mu_recon

    # manual training step
    def training_step(self, batch, batch_idx):
        dis1_opt, dis2_opt, dis_content_opt, encoder_opt, generator_opt = self.optimizers()

        fullres, lowres, domain = batch

        # log training images every opts.log_train_img_freq iterations
        half_size = fullres.size(0) // 2
        if batch_idx % self.opts.log_train_img_freq == 0:
            self.log_images(fullres[:1], fullres[half_size:half_size + 1], 'train')
            self.log_translated_images(fullres[:1], 'train')

        # update D, encoder, and generator every d_iter iterations and Dc else
        if (batch_idx + 1) % self.opts.d_iter == 0:
            # update Dc
            z_content, _ = self.encoder(lowres)
            dis_content_opt.zero_grad()
            pred_cls = self.dis_content.forward(z_content.detach())
            loss_dis_content = F.binary_cross_entropy_with_logits(pred_cls, domain)
            loss_dis_content.backward()
            nn.utils.clip_grad_norm_(self.dis_content.parameters(), 5)
            dis_content_opt.step()

            self.log_dict({'loss_dis/content': loss_dis_content}, prog_bar=True)
        else:

            # run forward pass once in training step
            z_content, z_attr, mu, log_var, z_random, fake_encoded_img, fake_A_a, fake_B_b, fake_A_recon, \
                fake_B_recon, fake_random_img, z_attr_random_a, z_attr_random_b, mu_recon = self.forward(batch)
                
            # update D
            dis1_opt.zero_grad()
            loss_dis1 = self.dis_loss(
                self.dis1, 1, fullres, fake_encoded_img.detach(), domain)
            loss_dis1.backward()
            dis1_opt.step()

            dis2_opt.zero_grad()
            loss_dis2 = self.dis_loss(
                self.dis2, 2, fullres, fake_random_img.detach(), domain)
            loss_dis2.backward()
            dis2_opt.step()

            # update encoder and generator
            encoder_opt.zero_grad()
            generator_opt.zero_grad()

            loss_gen = self.gen_loss(
                fullres, domain, z_content, z_attr, mu, log_var, fake_encoded_img, fake_A_a, fake_B_b, fake_A_recon,
                fake_B_recon, fake_random_img, z_attr_random_a, z_attr_random_b, z_random, mu_recon
            )
            loss_gen = sum(loss_gen)
            loss_gen.backward()

            encoder_opt.step()
            generator_opt.step()

            self.log_dict({
                'loss/gen_enc_total': loss_gen,
                'loss_dis/total_1': loss_dis1,
                'loss_dis/total_2': loss_dis2,
            }, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        fullres, _, _ = batch

        # run forward pass once in training step
        # _, _, _, _, _, _, fake_A_recon, fake_B_recon, _, _, _ = self.forward(
        _, _, _, _, _, _, _, _, fake_A_recon, fake_B_recon, _, _, _, _ = self.forward(batch)

        # log metric
        # -- self and cross-cycle recon
        loss_l1_cc = torch.mean(torch.abs(
            fullres - torch.cat((fake_A_recon, fake_B_recon), 0))) * self.opts.lambda_rec

        self.log_dict({
            'l1_cc_loss/val': loss_l1_cc,
            'l1_cc_loss_val': loss_l1_cc,
        })

        # log images
        half_size = fullres.size(0) // 2
        if batch_idx % self.opts.log_val_img_freq == 0:
            self.log_images(fullres[:1], fullres[half_size:half_size + 1], 'val')
            self.log_translated_images(fullres[:1], 'val')

    def dis_loss(self, dis, dis_id, image, fake_img, domain):
        # calculate loss for one discriminator
        pred_fake, pred_fake_cls = dis.forward(fake_img)
        pred_real, pred_real_cls = dis.forward(image)

        label_fake = torch.zeros_like(pred_fake)
        label_real = torch.ones_like(pred_real)

        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, label_fake)
        loss_real = F.binary_cross_entropy_with_logits(pred_real, label_real)

        loss_dis_adv = (loss_fake + loss_real) / 2
        loss_dis_cls = F.binary_cross_entropy_with_logits(pred_real_cls, domain)

        self.log_dict({
            f'loss_dis/adv_{dis_id}': loss_dis_adv,
            f'loss_dis/cls_{dis_id}': loss_dis_cls,
        })
        
        loss_dis_cls *= self.opts.lambda_cls  # 1.0
        loss_dis = loss_dis_adv + loss_dis_cls

        return loss_dis

    def gen_loss(self, image, domain, z_content, z_attr, mu, log_var, fake_encoded_img, fake_A_a, fake_B_b,
                 fake_A_recon, fake_B_recon, fake_random_img, z_attr_random_a, z_attr_random_b, z_random, mu_recon):
        # -- content adversarial loss for generator
        pred_cls = self.dis_content.forward(z_content)
        loss_adv_content = F.binary_cross_entropy_with_logits(pred_cls, 1 - domain)  # loss_G_GAN_content

        # -- adversarial loss for generator from dis1
        pred_fake, pred_fake_cls = self.dis1.forward(fake_encoded_img)
        label_real = torch.ones_like(pred_fake)
        loss_adv1 = F.binary_cross_entropy_with_logits(pred_fake, label_real)
        # -- classification (from dis1) (loss_G_cls)
        domain_swapped = torch.empty_like(domain)
        domain_swapped[:domain.shape[0]//2] = domain[domain.shape[0]//2:]
        domain_swapped[domain.shape[0]//2:] = domain[:domain.shape[0]//2:]
        loss_adv1_cls = F.binary_cross_entropy_with_logits(pred_fake_cls, domain_swapped)

        # -- self and cross-cycle recon
        loss_l1_self_recon = torch.mean(torch.abs(
            image - torch.cat((fake_A_a, fake_B_b), 0)))
        loss_l1_cc = torch.mean(torch.abs(
            image - torch.cat((fake_A_recon, fake_B_recon), 0)))

        # -- KL loss - z_c
        loss_kl_zc = self._l2_regularize(z_content)

        # -- KL loss - z_a
        kl_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
        loss_kl_za = (torch.sum(kl_element).mul_(-0.5)) / (kl_element.shape[0])
        # z_attr_randn = torch.randn_like(z_attr)
        # loss_kl_za = F.kl_div(z_attr, z_attr_randn, reduction='batchmean') 

        # -- adversarial loss for generator from dis2
        pred_fake, pred_fake_cls = self.dis2.forward(fake_random_img)
        label_real = torch.ones_like(pred_fake)
        loss_adv2 = F.binary_cross_entropy_with_logits(pred_fake, label_real)
        # -- classification (from dis2) (loss_G_cls)
        loss_adv2_cls = F.binary_cross_entropy_with_logits(pred_fake_cls, domain_swapped)

        # -- latent regression loss
        loss_z_L1_a = torch.mean(torch.abs(z_attr_random_a - z_random))
        loss_z_L1_b = torch.mean(torch.abs(z_attr_random_b - z_random))
        loss_z_L1 = loss_z_L1_a + loss_z_L1_b
        # -- latent regression loss in attribute space
        loss_l1_a_cc = F.l1_loss(mu_recon, mu)

        # -- contrastive loss in attribute space
        # label_domain = torch.argmax(domain, -1)
        # z_attr_normalized = F.normalize(mu.unsqueeze(1), dim=0)
        # loss_con_a = self.loss_contrastive(z_attr_normalized, labels=label_domain)

        # # -- contrastive loss in content space
        # z_content_dis = self.dis_c.forward(z_content).unsqueeze(1)
        # z_content_dis = F.normalize(z_content_dis, dim=0)
        # loss_con_disc = self.loss_contrastive(z_content_dis, labels=label_domain) * self.opts.lambda_cl_cont
        
        # -- paired loss with HistAuGAN
        # domain_histaugan = torch.eye(7)[torch.randint(0, 7, (image.shape[0],))].to(self.device)
        # with torch.no_grad():
        #     z_content_histaugan = self.histaugan.enc_c(image)
        #     mu_histaugan, log_var_histaugan = self.histaugan.enc_a(image, domain_histaugan)
        #     std = log_var_histaugan.mul(0.5).exp_()
        #     eps = torch.randn(std.size(0), std.size(1)).to(image.device)
        #     z_attr_histaugan = eps.mul(std).add_(mu_histaugan)
        #     paired_images = self.histaugan.gen(z_content_histaugan, z_attr_histaugan, domain_histaugan)
        # loss_paired = torch.mean(torch.abs(image - paired_images))

        self.log_dict({
            'loss/adv_content': loss_adv_content,
            'loss/adv_1': loss_adv1,
            'loss/adv_cls_1': loss_adv1_cls,
            'loss/adv_2': loss_adv2,
            'loss/adv_cls_2': loss_adv2_cls,
            'loss/l1_self_rec': loss_l1_self_recon,
            'loss/l1_cc': loss_l1_cc,
            'loss/kl_zc': loss_kl_zc,
            'loss/kl_za': loss_kl_za,
            'loss/l1_latent': loss_z_L1,
            'loss/l1_a_cc': loss_l1_a_cc,
            # 'loss/cl_a': loss_con_a,
            # 'loss/l1_paired': loss_paired,
            # 'loss_con_disc': loss_con_disc,
        })
        
        # loss scaling
        loss_adv_content *= self.opts.lambda_adv_c  # 1.0
        loss_adv1_cls *= self.opts.lambda_cls_G  # 5
        loss_l1_self_recon *= self.opts.lambda_rec  # 10
        loss_l1_cc *= self.opts.lambda_rec  # 10
        loss_kl_zc *= self.opts.lambda_kl_zc  # 0.01
        loss_kl_za *= self.opts.lambda_kl_za  # 0.01
        loss_adv2_cls *= self.opts.lambda_cls_G  # 5
        loss_z_L1 *= self.opts.lambda_z_L1  # 10
        loss_l1_a_cc *= self.opts.lambda_l1_a_cc  # 0
        # loss_paired *= self.opts.lambda_paired  # 10
        # loss_con_a *= self.opts.lambda_cl_attr  # 1.0

        return loss_adv_content, loss_adv1, loss_l1_self_recon, loss_l1_cc, loss_kl_zc, loss_kl_za,  \
            loss_adv2, loss_z_L1, loss_adv1_cls, loss_adv2_cls, loss_l1_a_cc  # , loss_paired  #, loss_l1_a_cc  #, loss_con_a  # , loss_con_disc  # 

    def log_images(self, real_a, real_b, phase, batch_idx=0):
        """
        log cross-cycle translation
        """
        self.eval()
        real_a_lowres = F.interpolate(real_a, size=(self.opts.lowres, self.opts.lowres))
        real_b_lowres = F.interpolate(real_b, size=(self.opts.lowres, self.opts.lowres))

        # get encodings
        real_img_lowres = torch.cat((real_a_lowres, real_b_lowres), 0)
        # z_content, z_attr = self.encoder(real_img_lowres)
        z_content, (mu, log_var) = self.encoder(real_img_lowres)
        z_content_a, z_content_b = torch.split(z_content, 1, dim=0)
        # z_attr_a, z_attr_b = torch.split(z_attr, 1, dim=0)
        
        # get global style encoding
        std = log_var.mul(0.5).exp_()
        # eps = torch.randn(std.size(0), std.size(1)).to(image.device)
        eps = torch.randn_like(std)
        z_attr = eps.mul(std).add_(mu)
        z_attr_a, z_attr_b = torch.split(z_attr, 1, dim=0)

        # get random z_attr
        z_random = torch.randn_like(z_attr_a)

        # first cross translation and reconstruction
        input_content_forA = torch.cat((z_content_a, z_content_a, z_content_a), 0)
        input_content_forB = torch.cat((z_content_b, z_content_b, z_content_b), 0)
        input_attr_forA = torch.cat((z_attr_b, z_attr_a, z_random), 0)
        input_attr_forB = torch.cat((z_attr_a, z_attr_b, z_random), 0)

        output_fakeA = self.generator(real_a.repeat((3, 1, 1, 1)), input_content_forA, input_attr_forA)
        output_fakeB = self.generator(real_b.repeat((3, 1, 1, 1)), input_content_forB, input_attr_forB)
        fake_A_attr_b, fake_A_attr_a, fake_A_random = torch.split(output_fakeA, z_content_a.size(0), dim=0)
        fake_B_attr_a, fake_B_attr_b, fake_B_random = torch.split(output_fakeB, z_content_a.size(0), dim=0)

        # get reconstructed encodings
        fake_encoded_img = torch.cat((fake_A_attr_b, fake_B_attr_a), 0)
        fake_encoded_img_lowres = F.interpolate(fake_encoded_img, size=(self.opts.lowres, self.opts.lowres))
        # z_content_recon, z_attr_recon = self.encoder(fake_encoded_img_lowres)
        # - local content encoding
        z_content_recon, (mu_recon, log_var_recon) = self.encoder(fake_encoded_img_lowres)
        z_content_recon_A, z_content_recon_B = torch.split(z_content_recon, 1, dim=0)
        # - global style encoding
        std_recon = log_var_recon.mul(0.5).exp_()
        # eps_recon = torch.randn(std_recon.size(0), std_recon.size(1)).to(image.device)
        eps_recon = torch.randn_like(std_recon)
        z_attr_recon = eps_recon.mul(std_recon).add_(mu_recon)
        z_attr_recon_b, z_attr_recon_a = torch.split(z_attr_recon, 1, dim=0)

        # second cross translation
        fake_A_recon = self.generator(fake_A_attr_b, z_content_recon_A, z_attr_recon_a)
        fake_B_recon = self.generator(fake_B_attr_a, z_content_recon_B, z_attr_recon_b)

        # original, first cc, first cc (random attribute), first cc (self-recon), second cc
        img = torch.cat((
            torch.cat((real_a.detach().cpu(), fake_A_attr_b[:1].detach().cpu(), fake_A_random[:1].detach().cpu(),
                       fake_A_attr_a[:1].detach().cpu(), fake_A_recon[:1].detach().cpu()), dim=3),
            torch.cat((real_b.detach().cpu(), fake_B_attr_a[:1].detach().cpu(), fake_B_random[:1].detach().cpu(),
                       fake_B_attr_b[:1].detach().cpu(), fake_B_recon[:1].detach().cpu()), dim=3)
        ), dim=2)

        self.logger.log_image(
            f'cross-cycle/{phase}', [(img / 2 + 0.5).squeeze(0)])
        self.train()

    def log_translated_images(self, image, phase, num_samples=3, batch_idx=0):
        self.eval()
        # get z_content
        image_lowres = F.interpolate(image, size=(self.opts.lowres, self.opts.lowres))
        # z_content, z_attr = self.encoder(image_lowres)
        z_content, (mu, log_var) = self.encoder(image_lowres)
        z_attr = mu
        # z_content = z_content.repeat(self.opts.num_domains, 1, 1, 1)

        # generate new histology image with same content as img
        translated_images = []
        for i in range(num_samples):
            z_attr = torch.randn_like(z_attr)
            out = self.generator(image, z_content, z_attr).detach()
            translated_images.append(out)

        img = torch.cat((image, *translated_images), dim=3)
        # grid = torchvision.utils.make_grid(
        #     torch.cat((image, *translated_images), dim=0), normalize=True, range=(-1, 1))
        self.logger.log_image(f'translated_images/{phase}', [(img / 2 + 0.5).squeeze(0)])
        self.train()

    def configure_optimizers(self):
        dis1_opt = torch.optim.Adam(self.dis1.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        dis2_opt = torch.optim.Adam(self.dis2.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        dis_content_opt = torch.optim.Adam(self.dis_content.parameters(
        ), lr=self.hparams.learning_rate / 2.5, betas=(0.5, 0.999), weight_decay=0.0001)
        encoder_opt = torch.optim.Adam(self.encoder.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        # enc_a_opt = torch.optim.Adam(self.enc_a.parameters(
        # ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)
        generator_opt = torch.optim.Adam(self.generator.parameters(
        ), lr=self.hparams.learning_rate, betas=(0.5, 0.999), weight_decay=0.0001)

        dis1_sch = networks.get_scheduler(dis1_opt, self.opts)
        dis2_sch = networks.get_scheduler(dis2_opt, self.opts)
        dis_content_sch = networks.get_scheduler(dis_content_opt, self.opts)
        encoder_sch = networks.get_scheduler(encoder_opt, self.opts)
        # enc_a_sch = networks.get_scheduler(enc_a_opt, self.opts)
        generator_sch = networks.get_scheduler(generator_opt, self.opts)

        return [dis1_opt, dis2_opt, dis_content_opt, encoder_opt, generator_opt], [dis1_sch, dis2_sch, dis_content_sch, encoder_sch, generator_sch]

    @staticmethod
    def _l2_regularize(mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


"""
taken from https://github.com/creotiv/hdrnet-pytorch (Feb 20, 2023)
pytorch implementation of https://github.com/google/hdrnet
Deep Bilateral Learning for Real-Time Image Enhancements, Siggraph 2017
"""
# -----------------------------------------------------------------------------------
# Network modules
# -----------------------------------------------------------------------------------


class Encoder(nn.Module):
    def __init__(self, nin=4, nout=3, params=None):
        super(Encoder, self).__init__()
        self.params = params
        self.nin = nin
        self.nout = nout

        lb = params['luma_bins']  # 8
        cm = params['channel_multiplier']  # 1
        sb = params['spatial_bin']  # 16
        bn = params['batch_norm']
        nsize = params['net_input_size']  # 256 | 128

        # splat features
        n_layers_splat = int(np.log2(nsize/sb))  # 4 | 3
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm*(2**i)*lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = cm*(2**i)*lb
        splat_ch = prev_ch
        
        # global features
        n_layers_global = int(np.log2(sb/4))  # 2
        self.global_features_conv = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(
                ConvBlock(prev_ch, cm*8*lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm*8*lb  # 64

        n_total = n_layers_splat + n_layers_global  # 6 = 4 + 2 | 5 = 3 + 2
        prev_ch = prev_ch * (nsize/2**n_total)**2  # 1024 = 64 * (256/2**6)**2
        self.global_features_fc = nn.ModuleList()
        self.global_features_fc.append(FC(prev_ch, 32*cm*lb, batch_norm=bn))  # 1024 -> 256
        self.global_features_fc.append(FC(32*cm*lb, 16*cm*lb, batch_norm=bn))  # 256 -> 128
        # self.global_features_fc.append(FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn))  # 128 -> 64
        self.global_features_fc_mean = nn.Sequential(*[FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn)])  # 128 -> 64
        self.global_features_fc_var = nn.Sequential(*[FC(16*cm*lb, 8*cm*lb, activation=None, batch_norm=bn)])  # 128 -> 64

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(
            ConvBlock(splat_ch, 8*cm*lb, 3, batch_norm=bn))
        self.local_features.append(
            ConvBlock(8*cm*lb, 8*cm*lb, 3, activation=None, use_bias=False))

    def forward(self, lowres_input):
        bs = lowres_input.shape[0]

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)
        splat_features = x

        # style-related global features
        for layer in self.global_features_conv:
            x = layer(x)
        x = x.view(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        # global_features = x
        global_features_mean = self.global_features_fc_mean(x)
        global_features_var = self.global_features_fc_var(x)

        # content-related local features
        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        # return local_features, global_features
        return local_features, (global_features_mean, global_features_var)


# class Generator(nn.Module):
#     def __init__(self, params=None):
#         super(Generator, self).__init__()
#         """
#         Initializes the Generator module.
        
#         Args:
#             params (dict): A dictionary of hyperparameters.
#         """
#         self.fusion = Fusion(params=params)
#         self.guide = GuideNN(params=params)
#         self.slicing = Slicing()
#         self.apply_coeffs = ApplyCoeffs()

#     def forward(self, fullres, local_features, global_features):
#         coeffs = self.fusion(local_features, global_features)
#         guide = self.guide(fullres)
#         slice_coeffs = self.slicing(coeffs, guide)
#         out = self.apply_coeffs(slice_coeffs, fullres)

#         return out
    
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, params=None):
        """
        Full-resolution path of HDRNet.
                
        Args:
            params (dict): A dictionary of hyperparameters.
        """
        super(Generator, self).__init__()
        
        # Initializes the sub-modules used by the Generator.
        self.fusion = Fusion(params=params)
        self.grayscale = transforms.Grayscale()
        self.guide = GuideNN(params=params)
        self.slicing = Slicing()
        self.apply_coeffs = ApplyCoeffs()

    def forward(self, fullres: torch.Tensor, local_features: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Generator module.
        
        Args:
            fullres (torch.Tensor): A tensor of shape (batch_size, num_channels, height, width)
                                    representing the full resolution input image.
            local_features (torch.Tensor): A tensor of shape (batch_size, num_local_features)
                                           representing the local features of the image.
            global_features (torch.Tensor): A tensor of shape (batch_size, num_global_features)
                                            representing the global features of the image.
        
        Returns:
            out (torch.Tensor): A tensor of shape (batch_size, num_channels, height, width)
                                representing the generated image.
        """
        # Computes the fusion of the local and global features.
        coeffs = self.fusion(local_features, global_features)
        
        # Computes the guide signal for the image.
        guide = self.guide(fullres)
        # guide = self.grayscale(fullres)
        
        # Slices the coefficients based on the guide signal.
        slice_coeffs = self.slicing(coeffs, guide)
        
        # Applies the sliced coefficients to the full resolution input image.
        out = self.apply_coeffs(slice_coeffs, fullres)

        return torch.tanh(out)  # to map ouput to [-1, 1]



class Fusion(nn.Module):
    def __init__(self, nin=4, nout=3, params=None):
        super(Fusion, self).__init__()
        self.nin = nin
        self.nout = nout
        self.params = params
        
        lb = params['luma_bins']  # 8
        cm = params['channel_multiplier']  # 1
        
        self.relu = nn.ReLU()
        
        # predicton
        self.conv_out = ConvBlock(8*cm*lb, lb*nout*nin, 1, padding=0, activation=None)

    def forward(self, local_features, global_features):
        bs = local_features.shape[0]
        lb = self.params['luma_bins']
        cm = self.params['channel_multiplier']

        fusion_grid = local_features
        fusion_global = global_features.view(bs, 8*cm*lb, 1, 1)
        fusion = self.relu(fusion_grid + fusion_global)

        x = self.conv_out(fusion)
        y = torch.stack(torch.split(x, self.nin*self.nout, 1), 2)

        return y


class GuideNN(nn.Module):
    def __init__(self, params=None):
        super(GuideNN, self).__init__()
        self.params = params

        self.conv1 = ConvBlock(
            3, params['guide_complexity'], kernel_size=1, padding=0, batch_norm=True)
        self.conv2 = ConvBlock(params['guide_complexity'], 1, kernel_size=1,
                               padding=0, activation=nn.Sigmoid)  # nn.Tanh nn.Sigmoid

    def forward(self, x):
        return self.conv2(self.conv1(x))  # .squeeze(1)


class Slicing(nn.Module):
    def __init__(self):
        super(Slicing, self).__init__()

    def forward(self, bilateral_grid, guidemap):
        bilateral_grid = bilateral_grid.permute(0, 3, 4, 2, 1)
        guidemap = guidemap.squeeze(1)
        # grid: The bilateral grid with shape (gh, gw, gd, gc).
        # guide: A guide image with shape (h, w). Values must be in the range [0, 1].
        
        coeefs = bilateral_slice(bilateral_grid, guidemap).permute(0, 3, 1, 2)
        return coeefs


class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()

    def forward(self, coeff, full_res_input):
        """
            most straightforward use case: take original RGB channels
            R = r1[0]*r2 + r1[1]*g2 + r1[2]*b3 +r1[3]
        """
        R = torch.sum(
            full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 9:10, :, :]
        G = torch.sum(
            full_res_input * coeff[:, 3:6, :, :], dim=1, keepdim=True) + coeff[:, 10:11, :, :]
        B = torch.sum(
            full_res_input * coeff[:, 6:9, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        return torch.cat([R, G, B], dim=1)


# -----------------------------------------------------------------------------------
# Network building blocks
# -----------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, use_bias=True, activation=nn.ReLU, batch_norm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(int(inc), int(
            outc), kernel_size, padding=padding, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

        if use_bias and not batch_norm:
            self.conv.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        # , mode='fan_out',nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

        if not batch_norm:
            self.fc.bias.data.fill_(0.00)
        # aka TF variance_scaling_initializer
        # , mode='fan_out',nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x

# -----------------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------------

# Reference:
# 1. (Yonglong Tian) Khosla P, Teterwak P, Wang C, et al. Supervised contrastive learning[J]. Advances in Neural Information Processing Systems, 2020, 33: 18661-18673.
# 2. (Kaiming He) He K, Fan H, Wu Y, et al. Momentum contrast for unsupervised visual representation learning[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020: 9729-9738.
# Difference:
# 1. use cosine_similarity to compute distance
# 2. support multi positive & negative samples
class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.10, contrast_mode='all',
                 base_temperature=0.10, contrastive_method='simclr'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        """
        Shape:
        :param x: (N, 1, C)
        :param y: (1, N, C)
        :return: (N, N)
        """
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            if self.contrastive_method == 'cl':
                mask = torch.eq(labels, labels.T).float().to(device)
            elif self.contrastive_method == 'infoNCE':
                loss = nn.functional.cross_entropy(features, labels)
                return loss * self.opts.lambda_cl_attr
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss