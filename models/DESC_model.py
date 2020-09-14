import torch
from .base_model import BaseModel
from . import networks
from utils.image_pool import ImagePool
from utils.util import *
import torch.nn.functional as F
from utils import dataset_util
import cv2
import numpy as np
import torchvision
import copy
import torch.nn as nn
import os
try:
    import cPickle as pickle
except:
    import pickle
from PIL import Image
import scipy
import random
from torch.autograd import Variable


class DESCModel(BaseModel):
    def name(self):
        return 'DESCModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        if is_train:
            parser.add_argument(
                '--lambda_S', type=float, default=50.0, help='weight for synthetic supervision')
            parser.add_argument(
                '--lambda_T', type=float, default=1.0, help='weight for semantic consistency')
            parser.add_argument('--lambda_Sm', type=float,
                                default=0.01, help='weight for depth smoothing')
            parser.add_argument('--lambda_IDT', type=float, default=100.0,
                                help='weight for image style transfer reconstruction')
            parser.add_argument('--lambda_GAN', type=float, default=1.0,
                                help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_St', type=float, default=50.0,
                                help='weight for stereo reconstruction loss')

        return parser

    def initialize(self, opt):
        # Object initialization
        self.total_steps = 0
        # Hyperparameters in Eq. (4)
        self.max_depth = 80

        BaseModel.initialize(self, opt)
        if self.isTrain:
            self.use_semantic_const = opt.use_semantic_const
            self.use_stereo = opt.use_stereo
            self.pretrain_semantic_module = opt.pretrain_semantic_module
            self.train_image_generator = opt.train_image_generator
            self.lambda_S = opt.lambda_S
            self.lambda_St = opt.lambda_St
            self.lambda_T = opt.lambda_T
            self.lambda_Sm = opt.lambda_Sm
            self.lambda_IDT = opt.lambda_IDT
            self.lambda_GAN = opt.lambda_GAN
            if self.use_semantic_const or self.pretrain_semantic_module:
                self.model_det = init_detections(opt)

            self.loss_names = ['', 'source_supervised', 'smooth']
            if self.train_image_generator:
                self.loss_names += ['image_generator']
            if self.use_semantic_const:
                self.loss_names += ['semantic_consistency']
            if self.use_stereo:
                self.loss_names += ['stereo']

        if self.isTrain:
            visual_names_src = ['src_img', 'src_real_depth']
            visual_names_src += ['src_gen_depth_s']
            visual_names_tgt = ['tgt_left_img',
                                'tgt_gen_depth_t', 'tgt_right_img']

            self.visual_names = visual_names_src
            self.visual_names += ['pred']
        else:
            self.visual_names = ['pred', 'img']

        if self.isTrain:
            if self.pretrain_semantic_module:
                self.model_names = ['G_Sem']
            else:
                self.model_names = ['G_Depth', '_s2t']
                if self.train_image_generator:
                    self.model_names += ['_Ds2t']
                if self.use_semantic_const:
                    self.model_names += ['G_Sem']
        else:
            self.model_names = ['G_Depth']

        if self.isTrain:
            if not self.pretrain_semantic_module:
                self.net_s2t = networks._ResGeneratorT2Net(3, 3, 64, 9, 'batch',
                                                           'PReLU', 0,
                                                           False, opt.gpu_ids).to(self.device)
                self.net_Ds2t = networks._MultiscaleDiscriminator(
                    3, 64, opt.n_layers_D, 1, 'batch', 'PReLU', opt.gpu_ids).to(self.device)
                self.netG_Depth = networks.init_net(networks.UNetGenerator(
                    norm='batch', input_nc=3), init_type='kaiming', gpu_ids=opt.gpu_ids)
                if not self.train_image_generator:
                    self.net_s2t.eval()
            if self.pretrain_semantic_module or self.use_semantic_const:
                self.netG_Sem = networks.init_net(networks.UNetGenerator(
                    norm='batch', input_nc=2, output_nc=1), init_type='kaiming', gpu_ids=opt.gpu_ids)
            self.fake_img_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSmooth = networks.SmoothLoss()
            self.criterionImgRecon = networks.ReconLoss()

            parameters = []
            if not self.pretrain_semantic_module:
                parameters = list(self.netG_Depth.parameters())

            if self.pretrain_semantic_module or self.use_semantic_const:
                parameters = parameters + list(self.netG_Sem.parameters())

            if self.use_semantic_const or self.pretrain_semantic_module:
                self.net_predict_height = networks.HeightPredictor().to(self.device)
                self.model_names += ['_predict_height']
                parameters = parameters + \
                    list(self.net_predict_height.parameters())
            self.scale_pred_l = torch.tensor(
                1.0, requires_grad=True, device='cuda')
            if self.train_image_generator:
                self.optimizer_G_task = torch.optim.Adam([{'params': self.net_s2t.parameters()},
                                                          {'params': [self.scale_pred_l], 'lr': opt.lr_task, 'betas': (
                                                              0.95, 0.999)},
                                                          {'params': parameters,
                                                           'lr': opt.lr_task, 'betas': (0.95, 0.999)}],
                                                         lr=opt.lr_trans, betas=(0.5, 0.9))
            else:
                self.optimizer_G_task = torch.optim.Adam([{'params': [self.scale_pred_l],
                                                           'lr': opt.lr_task, 'betas': (0.95, 0.999)},
                                                          {'params': parameters,
                                                           'lr': opt.lr_task, 'betas': (0.95, 0.999)}],
                                                         lr=opt.lr_trans, betas=(0.5, 0.9))

            if self.train_image_generator:
                parameters_D = list(self.net_Ds2t.parameters())
                self.optimizer_D = torch.optim.Adam(
                    parameters_D, lr=opt.lr_trans, betas=(0.5, 0.9))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G_task)
            if self.train_image_generator:
                self.optimizers.append(self.optimizer_D)

            self.optimizer_G_task.zero_grad()

            del parameters
        else:
            self.netG_Depth = networks.init_net(networks.UNetGenerator(
                norm='batch', input_nc=3), init_type='kaiming', gpu_ids=opt.gpu_ids)
            self.pretrain_semantic_module = False

    def set_input(self, input):

        if self.isTrain:
            self.src_real_depth = input['src']['depth'].to(self.device)
            self.src_img = input['src']['img'].to(self.device)
            self.src_original_img = input['src']['original_img']
            self.src_name = input['src']['name']
            self.src_focal = input['src']['focal']
            self.src_id = input['src']['id']
            self.tgt_left_img = input['tgt']['left_img'].to(self.device)
            self.tgt_original_left_img = input['tgt']['original_left_img']
            if 'right_img' in input['tgt']:
                self.tgt_right_img = input['tgt']['right_img'].to(self.device)
                self.tgt_original_right_img = input['tgt']['original_right_img']

            self.tgt_real_depth = input['tgt']['depth'].to(self.device)

            self.tgt_fb = input['tgt']['fb']
            self.tgt_focal = input['tgt']['focal']
            self.tgt_name = input['tgt']['name']
            self.tgt_id = input['tgt']['id']
            self.num = self.src_img.shape[0]
        else:
            self.img = input['left_img'].to(self.device)
            self.original_img = input['original_left_img'].to(self.device)
            self.depth = input['depth']
            self.fb = input['fb']
            self.focal = input['focal']
            self.data_name = input['name']
            self.id_img = input['id']

    def get_priors(self, depth_map, detections, fb, shape_imgs, is_target=False):

        # We initialize the depth pseudo label map
        depth_pl_map = -1 * torch.ones(shape_imgs).to(self.device)[:, :1, :, :]

        if len(depth_map.shape) < 4:
            depth_map = depth_map.unsqueeze(-2)

        # We upscale the depth map to the size of the images
        # used for detection, which are of higher res.
        # We could load directly the depth GT in a higher res,
        # but it should not affect much the results.
        res_depth_map = torch.nn.functional.interpolate(
            depth_map, size=shape_imgs[-2:], mode='nearest')
        res_depth_map = transform_depth(
            res_depth_map, to_meters=1, max_depth=self.max_depth)

        loss_prior = 0
        compute_loss_prior = 0
        total_elem = 0
        # First we iterate over images in batch
        for k in range(len(detections)):
            d = detections[k]
            fb_elem = fb[k]
            if d is not None:
                # We iterate over all detected instances per img
                num_detections_img = len(d.pred_boxes)
                for i in range(num_detections_img):

                    # We get the relevant elements from the instance
                    coor = d.pred_boxes[i].tensor[0]
                    mask = d.pred_masks[i].float().to(self.device)
                    pred_class = d.pred_classes[i].item()

                    # We get the coordinates
                    x1 = int(coor[0])
                    x2 = int(coor[2])
                    y1 = int(coor[1])
                    y2 = int(coor[3])

                    # In case the mask is blank for some reason
                    if mask[y1:y2, x1:x2].sum() == 0:
                        continue
                    mask_inst = mask[y1:y2, x1:x2].cuda(
                    ).unsqueeze(0).unsqueeze(0)
                    if mask_inst.shape[-2] == 0 or mask_inst.shape[-1] == 0:
                        continue

                    # We rescale the mask before inputting it to G_h
                    mask_inst = torch.nn.functional.interpolate(
                        mask_inst, size=[92, 308], mode='nearest')
                    # We transform the coordinates to a pytorch tensor
                    height_box = torch.FloatTensor(
                        coor).unsqueeze(0).to(self.device)

                    # We input the mask, coordinates and predicted class to G_h
                    # and we predict the height of the object
                    height_obj_pred = self.net_predict_height(mask_inst,
                                                              Variable(torch.LongTensor([pred_class])).to(
                                                                  self.device),
                                                              height_box)
                    # We use Equation 1 in the paper to compute an approximate depth
                    depth_approx = (
                        (fb_elem/(coor[3] - coor[1])) * height_obj_pred).float()
                    if not is_target:
                        # If it is the source dataset, we compute a loss using
                        # the ground_truth object height

                        # We first compute the object height GT using the ground-truth depth
                        # We get the instance depth using the instance mask
                        res_depth_map_mask = (
                            res_depth_map[k].squeeze() * mask)[y1:y2, x1:x2]
                        res_depth_map_mask = res_depth_map_mask[res_depth_map_mask > 0]
                        try:
                            # We use the median depth to compute the object height ground-truth
                            median_depth = res_depth_map_mask[res_depth_map_mask != 0].median(
                            )
                            if median_depth >= self.max_depth:
                                continue
                            # We apply h = D * H / f
                            height_obj = median_depth * (coor[3] - coor[1])/fb_elem
                        except:
                            # In some cases it breaks, so we handle this by skipping the instance
                            continue
                        loss_prior += self.lambda_S * \
                            (height_obj_pred - height_obj).abs().mean()
                        compute_loss_prior = 1
                        total_elem += 1
                    # We set the computed approx depth as the depth for
                    # the whole instance in the depth pseudo label map
                    depth_pl_map[k] = depth_approx * mask + (1-mask)*depth_pl_map[k]

        # We backpropagate the loss in the source dataset
        if not is_target and compute_loss_prior:
            loss_prior = loss_prior/total_elem
            loss_prior.backward()
        return depth_pl_map

    def get_detections(self, input_imgs, depth, fb, is_target=0):
        # We get the semantic annotations from the images
        # Including both semantic segmentation and height priors

        # First we compute semantic segmentation and instance detection
        # Using a pretrained panoptic model
        with torch.no_grad():
            detections = []
            semantic_seg = []
            for k in range(input_imgs.shape[0]):
                transformed_imgs = 255*(input_imgs[k, [2, 1, 0], :, :])
                inputs = {"image": transformed_imgs.detach().cuda(),
                          "height": input_imgs.shape[-2],
                          "width": input_imgs.shape[-1]}
                prediction = self.model_det.model([inputs])[0]
                instances = prediction["instances"].to('cpu')
                semantic_seg_inst = prediction['panoptic_seg'][0].clone()
                for elem in prediction['panoptic_seg'][1]:
                    semantic_seg_inst[elem['id'] ==
                                      prediction['panoptic_seg'][0]] = elem['category_id']
                semantic_seg.append(semantic_seg_inst.int())
                detections.append(instances)
            semantic_seg = torch.stack(semantic_seg, 0).to(self.device)
        # We get the height prior map and also compute the loss 
        # to train G_h if it is the source domain
        height_prior = self.get_priors(
            depth.clone(), detections, fb, input_imgs.shape, is_target=is_target)

        # The semantic annotations are in a higher resolution, so we downscale them
        height_prior = torch.nn.functional.interpolate(
            height_prior, size=depth.shape[-2:], mode='nearest')
        semantic_seg = torch.nn.functional.interpolate(
            semantic_seg.float().unsqueeze(1), size=depth.shape[-2:], mode='nearest')
        return height_prior, semantic_seg

    def forward(self):
        if self.isTrain:
            pass
        else:
            self.pred = self.netG_Depth(self.img)[-1]

    def backward_D_basic_list(self, netD, real, fake, detach=0):
        # Computes Least Squares loss given a discriminator
        # and real and fake images
        D_loss = 0
        for (real_i, fake_i) in zip(real, fake):
            # Real
            if detach:
                D_real = netD(real_i.detach())
            else:
                D_real = netD(real_i)
            # fake
            if detach:
                D_fake = netD(fake_i.detach())
            else:
                D_fake = netD(fake_i)

            for (D_real_i, D_fake_i) in zip(D_real, D_fake):
                D_loss += (torch.mean((D_real_i-1.0)**2) +
                           torch.mean((D_fake_i - 0.0)**2))*0.5
        return D_loss

    def backward_D_image(self, netD, detach=0):
        size = len(self.src_img_a)
        fake = []
        for i in range(size):
            # We save the fake images in a ImagePool as T2Net
            fake.append(self.fake_img_pool.query(self.src_img_a[i]))
        real = dataset_util.scale_pyramid(self.tgt_left_img, size)
        # Compute loss using discriminator
        loss_img_D = self.backward_D_basic_list(
            netD, real, fake, detach=detach)
        loss_img_D.backward()

    def backward_G(self):
        src_img_e = self.src_img.clone()
        batch_size = self.src_img.shape[0]
        if not self.pretrain_semantic_module:
            with torch.set_grad_enabled(self.train_image_generator):
                self.images = torch.cat([self.src_img, self.tgt_left_img])
                fake = self.net_s2t(self.images)
                self.src_img_a = []
                for i in range(1, len(fake)):
                    self.src_img_a.append(fake[i][:batch_size])
                self.src_img = self.src_img_a[-1]
                self.tgt_left_img_a = fake[-1][batch_size:]

        self.src_real_depth = transform_depth(
            self.src_real_depth, to_meters=1, max_depth=655.35)
        self.src_real_depth = self.src_real_depth.clamp(0, self.max_depth)
        self.src_real_depth = transform_depth(
            self.src_real_depth, to_meters=0, max_depth=self.max_depth)
        # =========================== synthetic ==========================
        if not self.pretrain_semantic_module:
            images_in = torch.cat(
                [self.src_img.clone(), self.tgt_left_img.clone()], 0)
            # Following T2Net we input the source and target batches
            # separately into the depth network. Thus, we freeze
            # the running stats for the source data, as in test time
            # we will only use target data.
            if not self.train_image_generator:
                freeze_running_stats(self.netG_Depth)
            self.out_s = self.netG_Depth(images_in[:batch_size])

            if not self.train_image_generator:
                freeze_running_stats(self.netG_Depth, unfreeze=1)
            self.out_t = self.netG_Depth(images_in[batch_size:])

        else:
            # We get the semantic segmentation information to pretrain the model
            _, semantic_seg_s = self.get_detections(
                self.src_original_img.clone(), self.src_real_depth,  self.src_focal)
            # We form the Sem. Seg. + Edges image from Section 3.2
            source_inp = torch.cat(
                [semantic_seg_s, get_edges(src_img_e)[:, :1]], 1)
            self.out_s = self.netG_Sem(source_inp.clone())

        self.loss_source_supervised = 0.0
        # Multi-scale depth loss
        self.src_gen_depth_s = self.out_s[-1]
        real_depths = dataset_util.scale_pyramid(
            self.src_real_depth.clone(), 4)
        for (gen_depth, real_depth) in zip(self.out_s, real_depths):
            self.loss_source_supervised += self.criterionL1(
                gen_depth, real_depth) * self.lambda_S

        # Below is the semantic consistency loss
        self.loss_semantic_consistency = 0.0
        if self.use_semantic_const:
            shape_imgs = self.src_img.shape[-2:]
            # We first get the predicted depth using height priors and the semantic segmentation
            # For that, we use a pretrained panoptic segmentation model and the original resolution images
            pseudo_src_depth, semantic_seg_s = self.get_detections(
                self.src_original_img.clone(), self.src_real_depth, self.src_focal)
            pseudo_tgt_depth, semantic_seg_t = self.get_detections(
                self.tgt_original_left_img.clone(), self.out_t[-1], self.tgt_focal, is_target=1)

            source_inp = torch.cat(
                [semantic_seg_s, get_edges(src_img_e)[:, :1]], 1)
            target_inp = torch.cat(
                [semantic_seg_t, get_edges(self.tgt_left_img)[:, :1]], 1)

            depth_map_sem = self.netG_Sem(
                torch.cat([source_inp, target_inp], 0))

            # This seems to behave better than the multiscale depth loss for this stage.
            self.loss_source_supervised += self.lambda_S * \
                (depth_map_sem[-1][:batch_size] -
                 self.src_real_depth).abs().mean()

            self.loss_semantic_consistency += self.lambda_T * \
                (depth_map_sem[-1][batch_size:] - self.out_t[-1]).abs().mean()

            mask = pseudo_tgt_depth != -1
            inst_loss = 0
            if mask.sum() > 0:
                pseudo_tgt_depth = pseudo_tgt_depth.clamp(
                    0, self.max_depth*self.scale_pred_l.item())
                pseudo_tgt_depth = pseudo_tgt_depth/self.scale_pred_l
                pseudo_tgt_depth = transform_depth(
                    pseudo_tgt_depth, to_meters=0, max_depth=self.max_depth)
                inst_loss = self.lambda_T * \
                    ((self.out_t[-1] - pseudo_tgt_depth)[mask].abs().mean())
            inst_loss = self.scale_pred_l * inst_loss
            self.loss_semantic_consistency += inst_loss

        l_imgs = dataset_util.scale_pyramid(self.tgt_left_img, 4)

        # smoothness
        # We only apply it to the target prediction of the main depth model, because
        # the depth estimated from the semantic->depth model tends to be smoother
        self.loss_smooth = 0.0
        if not self.pretrain_semantic_module:
            i = 0
            for (gen_depth, img) in zip(self.out_t, l_imgs):
                self.loss_smooth += self.criterionSmooth(
                    gen_depth, img) * self.lambda_Sm / 2**i
                i += 1

        # stereo consistency
        self.loss_stereo = 0.0
        if self.use_stereo:
            i = 0
            r_imgs = dataset_util.scale_pyramid(self.tgt_right_img, 4)
            for (l_img, r_img, gen_depth) in zip(l_imgs, r_imgs, self.out_t):
                loss, self.warp_tgt_img_t = self.criterionImgRecon(
                    l_img, r_img, gen_depth, self.tgt_fb / 2**(3-i), max_d=self.max_depth)
                self.loss_stereo += loss * self.lambda_St
                i += 1

        if self.train_image_generator:
            self.loss_image_generator = 0
            img_real = l_imgs
            size = len(img_real)
            D_fake = self.net_Ds2t(self.src_img_a[-1])
            self.loss_image_generator += self.lambda_IDT * \
                self.criterionL1(self.tgt_left_img_a, self.tgt_left_img)
            for D_fake_i in D_fake:
                self.loss_image_generator += self.lambda_GAN * \
                    torch.mean((D_fake_i - 1.0) ** 2)

        self.loss = self.loss_source_supervised + self.loss_smooth
        if self.use_semantic_const:
            self.loss = self.loss + self.loss_semantic_consistency
        if self.use_stereo:
            self.loss = self.loss + self.loss_stereo
        if self.train_image_generator:
            self.loss = self.loss + self.loss_image_generator

        self.loss_ = self.loss.detach()
        self.loss.backward()

    def optimize_parameters(self):
        # Optimization iteration
        if self.train_image_generator:
            self.set_requires_grad([self.net_Ds2t], False)
            self.optimizer_D.zero_grad()
        self.forward()
        self.optimizer_G_task.zero_grad()
        self.backward_G()
        self.optimizer_G_task.step()
        if self.train_image_generator:
            self.set_requires_grad([self.net_Ds2t], True)
            self.optimizer_D.zero_grad()
            self.backward_D_image(self.net_Ds2t, detach=1)
            self.optimizer_D.step()
