import torch
import numpy as np
import torch.nn as nn
from einops import rearrange,reduce,repeat
from einops.layers.torch import Rearrange

class DWConv(nn.Module):

    def __init__(self,in_dim,out_dim,kernel_size,stride=1,padding=0):

        super(DWConv,self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_dim,
                                   out_channels=in_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   groups=in_dim
                                   )
        
        self.pointwise = nn.Conv2d(in_channels=in_dim,
                                   out_channels=out_dim,
                                   kernel_size=1,
                                   stride=1
                                   )


    def forward(self,x):

        #x: (N,C,H,W)

        y = self.depthwise(x)
        y = self.pointwise(y)

        return y

              
class DWConvNeX(nn.Module):

    def __init__(self,in_dim,kernel_size,stride=1,padding=0):

        super(DWConvNeX,self).__init__()

        self.Dconv0 = nn.Conv2d(in_channels=in_dim,
                                out_channels=in_dim,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                groups=in_dim
                                )

        self.norm0 = nn.LayerNorm(in_dim,eps=1e-6)

        self.Pconv0 = nn.Conv2d(in_channels=in_dim,
                                out_channels=4*in_dim,
                                kernel_size=1,
                                stride=1,
                                )

        self.act0 = nn.LeakyReLU()

        self.proj_layer = nn.Conv2d(in_channels=in_dim*4,
                                   out_channels=in_dim,
                                   kernel_size=1,
                                   stride=1
                                   )

        self.trans_to = Rearrange("n c h w -> n h w c")
        self.trans_back = Rearrange("n h w c -> n c h w")

    def forward(self,x):

        #x : (N,in_dim,H,W)

        y = self.Dconv0(x) #(N,D,H,W)

        y = self.trans_to(y) #(N,H,W,D)

        y = self.norm0(y) #(N,H,W,D)

        y = self.trans_back(y) #(N,D,H,W)

        y = self.Pconv0(y) #(N,4*D,H,W)

        y = self.act0(y) #(N,4*D,H,W)

        y = self.proj_layer(y) #(N,in_dim,H,W)

        out = y + x #(N,in_dim,H,W)

        return out

class TinyEncoder(nn.Module):

    def __init__(self,in_dim):

        """
        params in_dim : pos_enc channel dim
        """
        super(TinyEncoder,self).__init__()

        self.embedding_layer = nn.Sequential(DWConv(in_dim=in_dim,out_dim=32,kernel_size=7,stride=1,padding=3),
                                            DWConvNeX(in_dim=32,kernel_size=7,stride=1,padding=3)
                                            )

        self.down1 = nn.Sequential(DWConv(in_dim=32,out_dim=32,kernel_size=5,stride=1,padding=2),
                                   nn.LeakyReLU(),
                                   DWConv(in_dim=32,out_dim=32,kernel_size=5,stride=1,padding=2),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

        self.down2 = nn.Sequential(DWConv(in_dim=64,out_dim=64,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   DWConv(in_dim=64,out_dim=64,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

        self.down3 = nn.Sequential(DWConv(in_dim=128,out_dim=128,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   DWConv(in_dim=128,out_dim=128,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

        self.down4 = nn.Sequential(DWConv(in_dim=256,out_dim=256,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   DWConv(in_dim=256,out_dim=256,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

        self.down5 = nn.Sequential(DWConv(in_dim=512,out_dim=512,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   DWConv(in_dim=512,out_dim=512,kernel_size=3,stride=1,padding=1),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )


    def forward(self,x):

        """
        params x: (N,C,H,W) 
        """

        y = self.embedding_layer(x) #(1,32,H,W)

        yout1 = self.down1(y) #(1,32,H,W) e.g.(1,32,384,512)

        yout2 = self.down2(yout1) #(1,128,H/4,W/4)  e.g.(1,128,96,128)

        yout3 = self.down3(yout2) #(1,256,H/8,W/8)  e.g.(1,256,48,64)

        yout4 = self.down4(yout3) #(1,512,H/16,W/16)  e.g.(1,512,24,32)

        yout5 = self.down5(yout4) #(1,512,H/32,W/32)  e.g.(1,512,12,16)

        return yout1,yout2,yout3,yout4,yout5

        

class Encoder(nn.Module):

    def __init__(self,in_dim):

        """
        params in_dim : batch_size * channels
        """
        super(Encoder,self).__init__()

        self.reshape_to = Rearrange("b c h w -> (b c) h w")

        self.embedding_layer = nn.Sequential(DWConv(in_dim=in_dim,out_dim=32,kernel_size=7,stride=1,padding=3),
                                            DWConvNeX(in_dim=32,kernel_size=7,stride=1,padding=3)
                                            )

        self.down1 = nn.Sequential(DWConvNeX(in_dim=32,kernel_size=5,stride=1,padding=2),
                                   DWConvNeX(in_dim=32,kernel_size=5,stride=1,padding=2),
                                   nn.Conv2d(in_channels=32,out_channels=64,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )


        self.down2 = nn.Sequential(DWConvNeX(in_dim=64,kernel_size=3,stride=1,padding=1),
                                   DWConvNeX(in_dim=64,kernel_size=3,stride=1,padding=1),
                                   nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

        self.down3 = nn.Sequential(DWConvNeX(in_dim=128,kernel_size=3,stride=1,padding=1),
                                   DWConvNeX(in_dim=128,kernel_size=3,stride=1,padding=1),
                                   nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

        self.down4 = nn.Sequential(DWConvNeX(in_dim=256,kernel_size=3,stride=1,padding=1),
                                   DWConvNeX(in_dim=256,kernel_size=3,stride=1,padding=1),
                                   nn.Conv2d(in_channels=256,out_channels=512,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

        self.down5 = nn.Sequential(DWConvNeX(in_dim=512,kernel_size=3,stride=1,padding=1),
                                   DWConvNeX(in_dim=512,kernel_size=3,stride=1,padding=1),
                                   nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,stride=1),
                                   nn.AvgPool2d(kernel_size=2)
                                   )

    def forward(self,x):

        """
        params x: (N,C,H,W) 
        """
        y = self.reshape_to(x) #(N*C,H,W)

        y = y.unsqueeze(0) #(1,N*C,H,W)

        y = self.embedding_layer(y) #(1,32,H,W) e.g.(1,32,384,512)

        dout1 = self.down1(y) #(1,64,H/2,W/2)  e.g.(1,64,192,256)

        dout2 = self.down2(dout1) #(1,128,H/4,W/4)  e.g.(1,128,96,128)

        dout3 = self.down3(dout2) #(1,256,H/8,W/8)  e.g.(1,256,48,64)

        dout4 = self.down4(dout3) #(1,512,H/16,W/16)  e.g.(1,512,24,32)

        dout5 = self.down5(dout4) #(1,512,H/32,W/32)  e.g.(1,512,12,16)

        return dout1,dout2,dout3,dout4,dout5


class Decoder(nn.Module):

    def __init__(self,tpos_dec_in_dim,dir_in_dim):

        """
        param tpos_in_dim : channel dim of encoded tpos 
        param dir_in_dim : channel dim of direction input
        """

        super(Decoder,self).__init__()

        self.up1 = nn.Sequential(nn.Conv2d(in_channels=512+tpos_dec_in_dim,out_channels=(512+tpos_dec_in_dim)*4,kernel_size=1,stride=1),
                                nn.LeakyReLU(),
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels=512+tpos_dec_in_dim,out_channels=512,kernel_size=1,stride=1),
                                DWConvNeX(in_dim=512,kernel_size=3,stride=1,padding=1),
                                DWConvNeX(in_dim=512,kernel_size=3,stride=1,padding=1),
                                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,stride=1),
                                nn.LeakyReLU()
                                )

        self.up2 = nn.Sequential(nn.Conv2d(in_channels=512+512+512,out_channels=512+512+512+512,kernel_size=1,stride=1),
                                nn.LeakyReLU(),
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels=512,out_channels=512,kernel_size=1,stride=1),
                                DWConvNeX(in_dim=512,kernel_size=3,stride=1,padding=1),
                                DWConvNeX(in_dim=512,kernel_size=3,stride=1,padding=1),
                                nn.Conv2d(in_channels=512,out_channels=256,kernel_size=1,stride=1),
                                nn.LeakyReLU()
                                )

        self.up3 = nn.Sequential(nn.Conv2d(in_channels=256+256+256,out_channels=256+256+256+256,kernel_size=1,stride=1),
                                nn.LeakyReLU(),
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1),
                                DWConvNeX(in_dim=256,kernel_size=3,stride=1,padding=1),
                                DWConvNeX(in_dim=256,kernel_size=3,stride=1,padding=1),
                                nn.Conv2d(in_channels=256,out_channels=128,kernel_size=1,stride=1),
                                nn.LeakyReLU()
                                )

        self.up4 = nn.Sequential(nn.Conv2d(in_channels=128+128+128,out_channels=128+128+128+128,kernel_size=1,stride=1),
                                nn.LeakyReLU(),
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels=128,out_channels=128,kernel_size=1,stride=1),
                                DWConvNeX(in_dim=128,kernel_size=3,stride=1,padding=1),
                                DWConvNeX(in_dim=128,kernel_size=3,stride=1,padding=1),
                                nn.Conv2d(in_channels=128,out_channels=64,kernel_size=1,stride=1),
                                nn.LeakyReLU()
                                )

        self.up5 = nn.Sequential(nn.Conv2d(in_channels=64+64+64,out_channels=64+64+64+64,kernel_size=1,stride=1),
                                nn.LeakyReLU(),
                                nn.PixelShuffle(upscale_factor=2),
                                nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1,stride=1),
                                DWConvNeX(in_dim=64,kernel_size=3,stride=1,padding=1),
                                DWConvNeX(in_dim=64,kernel_size=3,stride=1,padding=1),
                                nn.Conv2d(in_channels=64,out_channels=32,kernel_size=1,stride=1),
                                nn.LeakyReLU()
                                )

        #density
        self.fc_density = nn.Sequential(DWConvNeX(in_dim=32,kernel_size=3,stride=1,padding=1),
                                        nn.Conv2d(in_channels=32,out_channels=1,kernel_size=1,stride=1)
                                        )

        #color
        self.fc_feature = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=1,stride=1)
        self.rgb_layer =  nn.Sequential(DWConvNeX(in_dim=32+dir_in_dim,kernel_size=3,stride=1,padding=1),
                                        nn.Conv2d(in_channels=32+dir_in_dim,out_channels=(32+dir_in_dim)//2,kernel_size=1,stride=1),
                                        nn.LeakyReLU()
                                        )

        self.fc_rgb = nn.Conv2d(in_channels=(32+dir_in_dim)//2,out_channels=3,kernel_size=1,stride=1)

        self.transform_back = Rearrange("c h w -> h w c")
                

    def forward(self,enc_out,tpos_enc,dir_enc):

        """
        params enc_out : list [dout1,dout2,dout3,dout4,dout5] , dout : #(N,C,H,W)  e.g.(1,512,12,16)
        params tpos_enc: list [xout1,xout2,xout3,xout4,xout5] , xout : #(N,C,H,W)  e.g.(1,512,12,16)
        params dir_enc: (1,27,H,W) e.g. (1,27,384,512)
        """

        dout1,dout2,dout3,dout4,dout5 = enc_out
        xout1,xout2,xout3,xout4,xout5 = tpos_enc

        #up1
        y1 = torch.cat([dout5,xout5],dim=1) #(N,2*C,H,W)
        y1 = self.up1(y1) #(1,512,H/16,W/16) e.g.(1,512,24,32)

        #up2
        y2 = torch.cat([y1,dout4,xout4],dim=1)  #(N,3*C,H,W)  e.g.(1,512+512+512,24,32)
        y2 = self.up2(y2) #(1,256,H/8,W/8)  e.g.(1,256,48,64)

        #up3
        y3 = torch.cat([y2,dout3,xout3],dim=1) #(N,3*C,H,W)  e.g.(1,256+256+256,48,64)
        y3 = self.up3(y3) #(1,128,H/4,W/4)  e.g.(1,128,96,128)

        #up4
        y4 = torch.cat([y3,dout2,xout2],dim=1) #(N,3*C,H,W)  e.g.(1,128+128+128,96,128)
        y4 = self.up4(y4) #(1,64,H/2,W/2)  e.g.(1,64,192,256)

        #up5
        y5 = torch.cat([y4,dout1,xout1],dim=1) #(N,3*C,H,W)  e.g.(1,64+64+64,192,256)
        y5 = self.up5(y5) #(1,32,H,W)  e.g.(1,32,384,512)


        #density
        density = self.fc_density(y5) #(1,1,H,W)  e.g.(1,1,384,512)
        density = density.squeeze(0) #(1,H,W)
        
        #color
        rgb_feat = self.fc_feature(y5) #(1,32,H,W)  e.g.(1,32,384,512)
        rgb_feat = torch.cat([rgb_feat,dir_enc],dim=1) #(1,32+27,H,W)  e.g.(1,32+27,384,512)
        rgb_feat = self.rgb_layer(rgb_feat) #(1,16,H,W)  e.g.(1,16,384,512)
        rgb_feat = self.fc_rgb(rgb_feat)  #(1,3,H,W)  e.g.(1,3,384,512)
        rgb_feat = rgb_feat.squeeze(0) #(3,H,W)  e.g.(3,384,512)

        rgb_den  = torch.cat([rgb_feat,density],dim=0) #(4,H,W)  e.g.(4,384,512)
        rgb_den = self.transform_back(rgb_den) #(H,W,4)  e.g.(384,512,4)

        return rgb_den #(H,W,4)  e.g.(384,512,4)

class UNerf(nn.Module):

    def __init__(self,enc_in_dim,tpos_in_dim,dir_in_dim):

        super(UNerf,self).__init__()

        self.pos_enc_block = TinyEncoder(in_dim=tpos_in_dim)
        self.enc_block = Encoder(in_dim=enc_in_dim)
        self.dec_block = Decoder(tpos_dec_in_dim=512,dir_in_dim=dir_in_dim)

    def forward(self,img_data,tpos_enc,dir_enc):

        """
        params img_data: (N,C,H,W)
        params tpos_enc: (H, W, N_sample, 63) e.g.(384,512,128,63)
        params dir_enc: (H, W, N_sample, 27)  e.g. (384, 512, 128, 27)
        """

        #encode
        dout1,dout2,dout3,dout4,dout5 = self.enc_block(img_data)
        enc_out = [dout1,dout2,dout3,dout4,dout5]

        #decode
        _,_,N_sam,_ = tpos_enc.shape[0],tpos_enc.shape[1],tpos_enc.shape[2],tpos_enc.shape[3]

        yout = []

        for n in range(N_sam):
            
            #space time
            tpos_enc_x_in = tpos_enc[:,:,n,:] #(H, W, c)
            tpos_enc_x_in = rearrange(tpos_enc_x_in,"h w c -> c h w") #(c, H, W)
            tpos_enc_x_in = tpos_enc_x_in.unsqueeze(0) #(1, c, H, W)
            xout1,xout2,xout3,xout4,xout5 = self.pos_enc_block(tpos_enc_x_in) 
            tpos_out = [xout1,xout2,xout3,xout4,xout5]

            #direction
            dir_enc_x_in = dir_enc[:,:,n,:] #(H, W, c)
            dir_enc_x_in = rearrange(dir_enc_x_in,"h w c -> c h w") #(c, H, W)
            dir_enc_x_in = dir_enc_x_in.unsqueeze(0)  #(1, c, H, W)

            rgb_den = self.dec_block(enc_out,tpos_out,dir_enc_x_in) #(H,W,4)  e.g.(384,512,4)

            yout.append(rgb_den)

        yout = torch.stack(yout,dim=2) #(H,W,N_sample,4)

        return yout


if __name__ == "__main__":

    """
    model= UNerf(3*5,63,27)

    img_data = torch.randn(5,3,64,64)
    tpos_enc = torch.randn(64,64,64,63)
    dir_enc = torch.randn(64,64,64,27)

    yout = model(img_data,tpos_enc,dir_enc)
    """
    print("ok")

            




