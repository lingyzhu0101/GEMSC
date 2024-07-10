import os
import math
from decimal import Decimal

import utility
import IPython
import torch.nn.functional as F
import torch.nn as nn

import torchvision
import torch
from torch.autograd import Variable
from tqdm import tqdm
import pytorch_ssim
import torchvision
from PIL import Image
import numpy as np

import pywt
import torch_dct as dct
# https://github.com/zh217/torch-dct

def tensor_save_rgbimage(tensor, filename, cuda=False):
	if cuda:
		img = tensor.clone().cpu().clamp(0, 255).numpy()
	else:
		img = tensor.clone().clamp(0, 255).numpy()
	img = img.transpose(1, 2, 0).astype('uint8')
	img = Image.fromarray(img)
	img.save(filename)


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class vgg_v2(nn.Module):
    def __init__(self, vgg_model):
        super(vgg_v2, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

def vgg_loss(vgg, img, gt):
    mse = nn.MSELoss(size_average=True)
    img_vgg = vgg(img)
    gt_vgg = vgg(gt)
    return mse(img_vgg[0], gt_vgg[0]) + 0.6*mse(img_vgg[1], gt_vgg[1]) + 0.4*mse(img_vgg[2], gt_vgg[2]) + 0.2*mse(img_vgg[3], gt_vgg[3])

def vgg_init(vgg_loc):
    vgg_model = torchvision.models.vgg16(pretrained = False).cuda()
    vgg_model.load_state_dict(torch.load(vgg_loc))
    trainable(vgg_model, False)
    return vgg_model

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model = my_model
        self.loss = my_loss

        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()

        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        criterion_ssim = pytorch_ssim.SSIM(window_size = 11)
        criterion_mse = nn.MSELoss(size_average=True)
        criterion_tv = TVLoss()

        vgg_model = vgg_init('./pretrained/vgg16-397923af.pth')
        vgg = vgg_v2(vgg_model)
        vgg.eval()
        
        for batch, (lr, lr_gray, hr, _, idx_scale) in enumerate(self.loader_train):
            
            # prepare data
            # cA, (cH, cV, cD) = pywt.dwt2(lr_gray,"haar")
            # lr_cH = np.uint8(cH/np.max(cH)*255)
            # lr_cH = torch.from_numpy(lr_cH).float()
            # lr_cV = np.uint8(cV/np.max(cV)*255)
            # lr_cV = torch.from_numpy(lr_cV).float()
            # lr_cD = np.uint8(cD/np.max(cD)*255)
            # lr_cD = torch.from_numpy(lr_cD).float()
            # wave_image = torch.cat((lr_cH, lr_cV, lr_cD),1)
            # lr_wave_image = F.interpolate(wave_image, scale_factor=2, mode='bilinear', align_corners=True)
            # lr, lr_wave_image, hr  = self.prepare(lr, lr_wave_image, hr) # [4, 3, 400, 600]
            
            lr_wave_image = dct.dct(lr_gray)
            lr_wave_image = lr_wave_image.repeat(1,3,1,1)
            lr_wave_image = Variable(lr_wave_image)
            # Prepare data
            lr, lr_wave_image, hr = self.prepare(lr, lr_wave_image, hr) # here replace canny with high frequency
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            lr = lr/255.0
            hr = hr/255.0
            lr_wave_image = lr_wave_image/255.0
 
            [b, c, h, w] = hr.shape
            
            input_image = torch.cat((lr,lr_wave_image),1)
            # self.count_network_parameters(self.model, 'GMESC_DILATED')
            
            phr1, mask2 = self.model(input_image, 3)

            hr4 = hr[:, :, 0::4, 0::4]
            hr2 = hr[:, :, 0::2, 0::2]
            hr1 = hr 

            rect_loss = criterion_ssim(phr1, hr1) # +  criterion_ssim(phr2, hr2) + criterion_ssim(phr4, hr4)
            percept_loss = vgg_loss(vgg, phr1, hr1)
            tv_loss = 1e-4*criterion_tv(mask2)
            full_loss = rect_loss + percept_loss + tv_loss
 
            if full_loss.item() < self.args.skip_threshold * self.error_last:
                full_loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, rect_loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}=\t{}+\t{}+\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    full_loss.item(),
                    rect_loss.item(),
                    percept_loss.item(),
                    tv_loss.item(),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        print(rect_loss.item())

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)

                for idx_img, (lr, lr_gray, hr, filename, _) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)

                    # prepare data
                    # cA, (cH, cV, cD) = pywt.dwt2(lr_gray,"haar")
                    # lr_cH = np.uint8(cH/np.max(cH)*255)
                    # lr_cH = torch.from_numpy(lr_cH).float()
                    # lr_cV = np.uint8(cV/np.max(cV)*255)
                    # lr_cV = torch.from_numpy(lr_cV).float()
                    # lr_cD = np.uint8(cD/np.max(cD)*255)
                    # lr_cD = torch.from_numpy(lr_cD).float()
                    # wave_image = torch.cat((lr_cH, lr_cV, lr_cD),1)
                    # lr_wave_image = F.interpolate(wave_image, scale_factor=2, mode='bilinear', align_corners=True)

                    lr_wave_image = dct.dct(lr_gray)
                    lr_wave_image = lr_wave_image.repeat(1,3,1,1)  
                    lr_wave_image = Variable(lr_wave_image)     
                    if not no_eval:
                        lr, lr_wave_image, hr = self.prepare(lr, lr_wave_image, hr)
                    else:
                        lr, = self.prepare(lr)
                    
                    lr = lr/255.0
                    hr = hr/255.0
                    lr_wave_image = lr_wave_image/255.0

                    [b, c, h, w] = hr.shape
                    n_map = torch.zeros(b, c, h, w).cuda()

                    input_image = torch.cat((lr, lr_wave_image),1)
                    phr1, mask2 = self.model(input_image, 3)
                    
                    #phr = utility.quantize(mask2*255, self.args.rgb_range)
                    phr = utility.quantize(phr1*255, self.args.rgb_range)
                    # phr = utility.quantize(lr_wave_image*255, self.args.rgb_range)
                    lr = utility.quantize(lr*255, self.args.rgb_range)
                    hr = utility.quantize(hr*255, self.args.rgb_range)

                    save_list = [hr, lr, phr, lr]

                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            phr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale, epoch)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def count_network_parameters(self, model, name):
        """Count network parameters only for requires_grad=True."""
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        N = sum([np.prod(p.size()) for p in parameters])
        print("=== The number of parameters of the above model [{}] is [{}] or [{:>.4f}M] ===".format(name, N, N / 1e6))    

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

