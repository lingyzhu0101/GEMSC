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


        return input, mask1, mask2

      
class DilaLayer(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1 ,L=32):
        super(DilaLayer, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
       
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=1, dilation = 1,groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
            
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=2, dilation = 2, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
            
        self.convs.append(nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=stride, padding=4, dilation = 4, groups=G),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=False)
        ))
                
       
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        #print("--------0",x.shape) # [32 ,32, 256, 256]
        for i, conv in enumerate(self.convs):
            #print("-------times",i)
            fea = conv(x).unsqueeze_(dim=1)
            #print("--------1",fea.shape) # [32, 1,32, 256, 256] 
            if i == 0:
                feas = fea
                #print("--------2",feas.shape) # [32, 1,32, 256, 256] -- > # [32, 2,32, 256, 256] -- > # [32, 3,32, 256, 256]
            else:
                feas = torch.cat([feas, fea], dim=1)
                #print("--------3",feas.shape) # [32 , 2, 32, 256, 256]  --> [32 , 3, 32, 256, 256]
        fea_U = torch.sum(feas, dim=1)
        #print("--------4",fea_U.shape) # [32, 32, 256, 256]
        fea_s = fea_U.mean(-1).mean(-1)
        #print("--------5",fea_s.shape) #  [32, 32] 
        fea_z = self.fc(fea_s)
        #print("--------6",fea_z.shape) #  [32, 32]
        for i, fc in enumerate(self.fcs):
            #print("-------next times",i)
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
                #print("--------7",attention_vectors.shape) #   [32, 1,32]
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
                #print("--------8",attention_vectors.shape) #   [32, 2,32] -- >  [32, 3,32]
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        #print("--------9",attention_vectors.shape) # [32, 3, 32, 1, 1]
        fea_v = (feas * attention_vectors).sum(dim=1) # [32 , 3, 32, 256, 256] *  [32, 3, 32, 1, 1]
        #print("--------10",fea_v.shape)# [32, 32, 256, 256]      
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
        
        self.shortcut = nn.Sequential() # explore more illumination layers/contrast layers
        
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)




class MSRN_ATT(nn.Module):
    def __init__(self, args, conv=common.default_conv):
    #def __init__(self, conv= common.default_conv):
        super(MSRN_ATT, self).__init__()
        n_feats = 64
        n_blocks = 3
        kernel_size = 3
        act = nn.ReLU(True)
        self.n_blocks = n_blocks
        
        # define head module
        modules_head = [conv(6, n_feats, kernel_size)] # here we input image + guidance map
        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(DetailBlcok(n_feats=n_feats))
        # define tail module
        modules_tail = [
            nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1),
            conv(n_feats, n_feats, kernel_size),
            conv(n_feats, 3, kernel_size)] # 

        
        self.lstmconv = nn.Sequential(LSTMCONV()) 
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        
        x, mask1, mask2 = self.lstmconv(x)
        #print("----------0", x.shape, mask2.shape)  # [32, 3, 256, 256]        
        x = torch.cat((x, mask2), 1)
        #print("----------1", x.shape)  # [32, 3, 256, 256]
        x = self.head(x) 
        res = x
        #print("----------1", res.shape) # [32, 64, 256, 256]
        MSRB_out = []
        x = self.body(x)
        #print("----------2", x.shape) # [32, 64, 256, 256]
        MSRB_out.append(x)
        MSRB_out.append(res)

        res = torch.cat(MSRB_out,1)
        #print("----------3", res.shape) # [32, 128, 256, 256]
        x = self.tail(res)
        #print("----------4", x.shape)# [32, 3, 256, 256]
        return x, mask1, mask2 



if __name__ == "__main__":
    m = MSRN_ATT()
    print(m)
    input = torch.randn(32, 3, 256, 256)
    C_output = m(input) 
    
    
    
    
    
     
