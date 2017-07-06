#!/usr/bin/env python
# encoding: utf-8

import torch.nn as nn


# input: batch_size * nc * 64 * 64
# output: batch_size * k * 1 * 1
class Encoder(nn.Module):
    def __init__(self, isize, nc, k=100, ndf=64):
        super(Encoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        # input is nc x isize x isize
        main = nn.Sequential()
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, k, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


# input: batch_size * k * 1 * 1
# output: batch_size * nc * image_size * image_size
class Decoder(nn.Module):
    def __init__(self, isize, nc, k=100, ngf=64):
        super(Decoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial.{0}-{1}.convt'.format(k, cngf), nn.ConvTranspose2d(k, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf), nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf), nn.ReLU(True))

        csize = 4
        while csize < isize // 2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid.{0}.relu'.format(cngf // 2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc), nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())

        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


def grad_norm(m, norm_type=2):
    total_norm = 0.0
    for p in m.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)
