# Multi-scale Residual Network for Image 
from model import common
#import common
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return MSRN_ATT(args)

class LSTMCONV(nn.Module):
    def __init__(self):
        super(LSTMCONV, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
            )
            

    def forward(self, origin_input):
        # split data
        input, lr_wave_image = torch.chunk(origin_input, 2, dim=1)
        
        # initialize 
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask0 = torch.ones(batch_size, 3, row, col).cuda() / 2. # define three layers
        h = torch.zeros(batch_size, 32, row, col).cuda()
        c = torch.zeros(batch_size, 32, row, col).cuda()

        
        # First time lstm using lr
        x = torch.cat((input, mask0), 1)

        x = self.det_conv0(x)
        resx = x
        x = F.relu(self.det_conv1(x) + resx)
        resx = x
        x = F.relu(self.det_conv2(x) + resx)
        resx = x
        x = F.relu(self.det_conv3(x) + resx)
        resx = x
        x = F.relu(self.det_conv4(x) + resx)
        resx = x
        x = F.relu(self.det_conv5(x) + resx)
        x = torch.cat((x, h), 1)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        mask1 = self.det_conv_mask(h)

        # second time lstm using lr_wave_image
        x = torch.cat((lr_wave_image, mask1), 1)
        x = self.det_conv0(x)
        resx = x
        x = F.relu(self.det_conv1(x) + resx)
        resx = x
        x = F.relu(self.det_conv2(x) + resx)
        resx = x
        x = F.relu(self.det_conv3(x) + resx)
        resx = x
        x = F.relu(self.det_conv4(x) + resx)
        resx = x
        x = F.relu(self.det_conv5(x) + resx)
        x = torch.cat((x, h), 1)
        i = self.conv_i(x)
        f = self.conv_f(x)
        g = self.conv_g(x)
        o = self.conv_o(x)
        c = f * c + i * g
        h = o * torch.tanh(c)
        mask2 = self.det_conv_mask(h)

        return input, mask2

      
class DilaLayer(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1 ,L=32):
        super(DilaLayer, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])

        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
            
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=5, stride=stride, padding=2, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
            
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=9, stride=stride, padding=4, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
       
        # self.convs.append(nn.Sequential(
        #     nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, dilation = 1,groups=G),
        #     nn.BatchNorm2d(features),
        #     nn.ReLU(inplace=False)
        # ))
            
        # self.convs.append(nn.Sequential(
        #     nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=2, dilation = 2, groups=G),
        #     nn.BatchNorm2d(features),
        #     nn.ReLU(inplace=False)
        # ))
            
        # self.convs.append(nn.Sequential(
        #     nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=4, dilation = 4, groups=G),
        #     nn.BatchNorm2d(features),
        #     nn.ReLU(inplace=False)
        # ))
                
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
          
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1) 
        return fea_v 



class DetailBlcok(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(DetailBlcok, self).__init__()
        M = 3
        G = 8
        r = 2
        WH = n_feats
        out_features = n_feats
        
        if n_feats is not None:
            mid_features = int(n_feats/2)
        
        self.feas = nn.Sequential(
            nn.Conv2d(n_feats, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            DilaLayer(mid_features, WH, M, G, r, stride=1, L=32),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        self.shortcut = nn.Sequential() 
        
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)




class SFTLayer(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SFTLayer, self).__init__()

        self.SFT_scale_feas = nn.Sequential(
            nn.Conv2d(32, 32,   1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32 *2, 1, padding=0, stride=1)
        )
            
        self.SFT_shift_feas = nn.Sequential(
            nn.Conv2d(32, 32,   1, padding=0, stride=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32 *2, 1, padding=0, stride=1)
        )

    def forward(self, input_mask):
        #print("-----------------------------0000000",input_mask.shape)
        scale = self.SFT_scale_feas(input_mask)
        #print("-----------------------------0000001",scale.shape)
        shift = self.SFT_shift_feas(input_mask)
        #print("-----------------------------0000002",shift.shape)
        scale_shift = torch.cat((scale, shift),1)
        #print("-----------------------------0000003",scale_shift.shape)
        return scale_shift


class SFTBlcok(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(SFTBlcok, self).__init__()
        M = 3
        G = 8
        r = 2
        WH = 64
     
        self.feas = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding =1), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False), 
            nn.Conv2d(32, 32, 1), 
            nn.BatchNorm2d(32), 
            SFTLayer(32, WH, M, G, r, stride=1, L=32)
        ) 
        
    def forward(self, input_mask):
        #print("----------------------input_mask",input_mask.shape )    
        scale_shift = self.feas(input_mask)
        #print("-----------------scale_shift",scale_shift.shape)
    
        return scale_shift




class MSRN_ATT(nn.Module):
    def __init__(self, args, conv=common.default_conv):
    #def __init__(self, conv= common.default_conv):
        super(MSRN_ATT, self).__init__()
        n_feats = 64
        n_blocks = 3
        kernel_size = 3
        act = nn.ReLU(True)
        self.n_blocks = n_blocks

        self.lstmconv = nn.Sequential(LSTMCONV()) 
        # define head module
        modules_head = [conv(6, 64, 3)] # here we input image + guidance map
        # define body module
        modules_body_1 = [DetailBlcok(n_feats=64)]
        modules_map_1 =  [SFTBlcok(n_feats=64)]
      
        modules_body_2 = [DetailBlcok(n_feats=64)]
        modules_map_2 =  [SFTBlcok(n_feats=64)]
        
        modules_body_3 = [DetailBlcok(n_feats=64)]
        modules_map_3 =  [SFTBlcok(n_feats=64)]

        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, 3, kernel_size)] # 
            

        self.head = nn.Sequential(*modules_head)        
        self.map_1 = nn.Sequential(*modules_map_1)
        self.map_2 = nn.Sequential(*modules_map_2)
        self.map_3 = nn.Sequential(*modules_map_3)

        self.body_1 = nn.Sequential(*modules_body_1)
        self.body_2 = nn.Sequential(*modules_body_2)
        self.body_3 = nn.Sequential(*modules_body_3)
        
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x, mask2 = self.lstmconv(x)
        # print("----------0", x.shape, mask2.shape)  # [32, 3, 256, 256]      
        x = torch.cat((x, mask2), 1)  
        # print("----------1", x.shape) # [32, 64, 256, 256]

        x_head = self.head(x) 
        res_head = x_head
        
        

        # first time input  
        scale_shift_1 = self.map_1(mask2)
        scale_1, shift_1 = torch.chunk(scale_shift_1, 2, dim=1)
        
        x_1_original = self.body_1(x_head)
        x_1 = scale_1* x_1_original + shift_1
        x_1 = x_1 + x_1_original

        # second time input 
        scale_shift_2 = self.map_2(mask2)
        scale_2, shift_2 = torch.chunk(scale_shift_2, 2, dim=1)
        
        x_2_original = self.body_2(x_1)
        x_2 = scale_2* x_2_original + shift_2
        x_2 = x_2 + x_2_original
        

        # third time input 
        scale_shift_3 = self.map_3(mask2)
        scale_3, shift_3 = torch.chunk(scale_shift_3, 2, dim=1)
        
        x_3_original = self.body_3(x_2)
        x_3 = scale_3* x_3_original + shift_3
        x_3 = x_3 + x_3_original

        MSRB_out = []
        #print("----------2", x_3.shape) # [32, 64, 256, 256]
        MSRB_out.append(x_3)
        MSRB_out.append(res_head)

        res = torch.cat(MSRB_out,1)
        #print("----------3", res.shape) # [32, 128, 256, 256]
        out = self.tail(res)
        # print("----------4", x.shape)# [32, 3, 256, 256]
        return out, mask2 




if __name__ == "__main__":
    m = MSRN_ATT()
    print(m)
    input = torch.randn(32, 6, 256, 256)
    C_output = m(input) 
    
   
  
    
    
    
     
