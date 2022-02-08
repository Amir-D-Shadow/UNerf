import os, random, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import imageio
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose,Resize,ToTensor
import torchvision.transforms as transform
from einops import rearrange
from PIL import Image as Image

# utlities
from helper.utils.pos_enc import encode_position
from helper.utils.volume_op import volume_rendering, volume_sampling_ndc
from helper.utils.comp_ray_dir import comp_ray_dir_cam_fxfy

from helper.utils.lie_group_helper import convert3x4_4x4
from helper.utils.pose_utils import create_spiral_poses
from helper.loss_utils import *

from module import *

#setting
scene_name = "room1"
img_dir = f"{os.getcwd()}/data/{scene_name}"
model_weight_dir = f"{os.getcwd()}/model_weights/{scene_name}"

load_model = False
N_EPOCH = 1000  # set to 1000 to get slightly better results. we use 10K epoch in our paper.
EVAL_INTERVAL = 50  # render an image to visualise for every this interval.

device = torch.device("cuda:6")
device_ids = [6,7,8,9]
num_of_device = len(device_ids)

SSIM_loss = SSIM(size_average=True)
SSIM_loss = nn.DataParallel(SSIM_loss,device_ids=device_ids)
SSIM_loss.to(device)

#load data
def load_imgs(image_dir):
    img_names = np.array(sorted(os.listdir(image_dir)))  # all image names
    img_paths = [os.path.join(image_dir, n) for n in img_names]
    N_imgs = len(img_paths)

    img_list = []
    for p in img_paths:
        img = imageio.imread(p)[:, :, :3]  # (H, W, 3) np.uint8
        img = Image.fromarray(img).resize((512,384),Image.BILINEAR) #resize (512,384)
        img_list.append(img)
    img_list = np.stack(img_list)  # (N, H, W, 3)
    img_list = torch.from_numpy(img_list).float() / 255  # (N, H, W, 3) torch.float32
    H, W = img_list.shape[1], img_list.shape[2]
    
    results = {
        'imgs': img_list,  # (N, H, W, 3) torch.float32
        'img_names': img_names,  # (N, )
        'N_imgs': N_imgs,
        'H': H,
        'W': W,
    }
    return results

def load_data(image_path):

    folder = os.listdir(image_path) #list
    folder.sort(key = lambda x : int(x))
    TSteps = len(folder)

    image_data = {}

    image_data["TSteps"] = TSteps

    for t in range(TSteps):

        path = f"{image_path}/{folder[t]}/images"

        image_info = load_imgs(path)

        image_data[t] = image_info["imgs"] # (N, H, W, 3)
        image_data["N_IMGS"] = image_info['N_imgs']
        image_data["H"] = image_info['H']
        image_data["W"] = image_info['W']

    return image_data

image_data = load_data(img_dir)

#imgs = image_data['imgs']  # (N, H, W, 3) torch.float32
N_IMGS = image_data['N_IMGS'] 
H = image_data['H']
W = image_data['W']
TSteps = image_data["TSteps"]

print('Loaded {0} imgs, resolution {1} x {2}'.format(N_IMGS*TSteps, H, W))


#Learn Focal
class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad):
        super(LearnFocal, self).__init__()
        self.H = 500.0#H
        self.W = 500.0#W
        self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
        self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )

    def forward(self):
        # order = 2, check our supplementary.
        fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
        return fxfy

#learn pose
def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R


def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t):
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)
        return c2w


#set ray parameter
class RayParameters():

    def __init__(self):

      self.Win_H = 32 * num_of_device
      self.Win_W = 32
      self.NEAR, self.FAR = 0.0, 1.0  # ndc near far
      self.N_SAMPLE = 64  # samples per ray 128
      self.POS_ENC_FREQ = 10  # positional encoding freq for location
      self.DIR_ENC_FREQ = 4   # positional encoding freq for direction
      self.TSteps = TSteps

ray_params = RayParameters()


#set render function
def model_render_image(IMGS_input,c2w, rays_cam, t_vals, ray_params, H, W, fxfy, nerf_model,perturb_t, sigma_noise_std):
    """
    :param IMGS_input : (N,C,H,W)               image data for N cams
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param perturb_t:   True/False              perturb t values.
    :param sigma_noise_std: float               add noise to raw density predictions (sigma).
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """
    # KEY 2: sample the 3D volume using estimated poses and intrinsics online.
    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, ray_params.NEAR,
                                                                     ray_params.FAR, H, W, fxfy, perturb_t)

    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=ray_params.POS_ENC_FREQ, inc_input=True)

    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
    dir_enc = encode_position(ray_dir_world, levels=ray_params.DIR_ENC_FREQ, inc_input=True)  # (H, W, 27)
    dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, ray_params.N_SAMPLE, -1)  # (H, W, N_sample, 27)

    # inference rgb and density using position and direction encoding.
    rgb_density = nerf_model(IMGS_input,pos_enc, dir_enc)  # (H, W, N_sample, 4)

    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn=torch.sigmoid)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'depth_map': depth_map,  # (H, W)
    }

    return result


def train_one_epoch(image_data, H, W, ray_params, opt_nerf, opt_focal,opt_pose, nerf_model, focal_net, pose_param_net):

    nerf_model.train()
    focal_net.train()
    pose_param_net.train()

    t_vals = torch.linspace(ray_params.NEAR, ray_params.FAR, ray_params.N_SAMPLE, device=device)  # (N_sample,) sample position 
    ssim_loss_epoch = []
    psnr_loss_epoch = []
    total_loss_epoch = []


    t_list = [t for t in range(ray_params.TSteps)]
    random.shuffle(t_list)

    #set up
    num_rows = H // ray_params.Win_H
    num_cols = W // ray_params.Win_W

    for t in t_list:

        #get imgs
        IMGS = image_data[t]  # (N,H,W,3)

        # shuffle the training imgs
        ids = np.arange(N_IMGS)
        np.random.shuffle(ids)

        #set up row id and col id
        row_list = [( row_i * ray_params.Win_H , (row_i + 1) * ray_params.Win_H ) for row_i in range(num_rows)]
        random.shuffle(row_list)
        col_list = [( col_j * ray_params.Win_W , (col_j + 1) * ray_params.Win_W ) for col_j in range(num_cols)]
        random.shuffle(col_list)

        for i in ids:

            #render image by patch
            for row_id in row_list:

                row_start , row_end = row_id

                for col_id in col_list:

                    col_start , col_end = col_id

                    #set up
                    fxfy = focal_net()

                    # KEY 1: compute ray directions using estimated intrinsics online.
                    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
                    img = IMGS[i]  # (H, W, 3)
                    c2w = pose_param_net(i)  # (4, 4)

                    # crop 32x32 pixel on an image and their rays for training.
                    IMGS_input = rearrange( IMGS[:,row_start:row_end,col_start:col_end,:] , "b h w c -> b c h w") # (N,H,W,3) -> (N,3,H,W)
                    IMGS_input = IMGS_input.to(device) #(N,3,N_select_rows, N_select_cols)

                    ray_selected_cam = ray_dir_cam[row_start:row_end,col_start:col_end,:]  # (N_select_rows, N_select_cols, 3)

                    img_selected = img[row_start:row_end,col_start:col_end,:] # (N_select_rows, N_select_cols, 3)
                    img_selected = img_selected.to(device) # (N_select_rows, N_select_cols, 3)

                    # render an image using selected rays, pose, sample intervals, and the network
                    render_result = model_render_image(IMGS_input,c2w, ray_selected_cam, t_vals, ray_params,H, W, fxfy, nerf_model, perturb_t=True, sigma_noise_std=0.0)

                    #calculate loss
                    rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)
                    depth_rendered = render_result['depth_map'] # (N_select_rows, N_select_cols)
                    disp =  torch.reciprocal(depth_rendered) # (N_select_rows, N_select_cols)
                    #L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image

                    #l1 loss
                    rgb_l1_loss = F.l1_loss(rgb_rendered,img_selected)
                    rgb_l1_loss = rgb_l1_loss.mean()

                    #ssim
                    ssim_syn = rearrange( rgb_rendered.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)
                    ssim_tgt = rearrange( img_selected.unsqueeze(0), "b h w c -> b c h w") # (1,N_select_rows, N_select_cols, 3) -> (1,3,N_select_rows, N_select_cols)

                    rgb_ssim_loss =  1 - SSIM_loss(ssim_syn,ssim_tgt)
                    ssim_loss_epoch.append(rgb_ssim_loss.clone().detach())

                    #edge_aware_loss
                    EAL_loss = edge_aware_loss(img_selected,disp)

                    #sum the loss
                    total_loss = rgb_l1_loss + rgb_ssim_loss + EAL_loss
                    total_loss_epoch.append(total_loss.clone().detach())

                    #L2_loss.backward()
                    total_loss.backward()
                    opt_nerf.step()
                    opt_focal.step()
                    opt_pose.step()
                    opt_nerf.zero_grad()
                    opt_focal.zero_grad()
                    opt_pose.zero_grad()

                    with torch.no_grad():

                        psnr_loss = psnr(rgb_rendered,img_selected)
                        psnr_loss_epoch.append(psnr_loss)

                    

    total_loss_epoch_mean = torch.stack(total_loss_epoch).mean().item()
    ssim_loss_epoch_mean = torch.stack(ssim_loss_epoch).mean().item()
    psnr_loss_epoch_mean = torch.stack(psnr_loss_epoch).mean().item()

    return [psnr_loss_epoch_mean,ssim_loss_epoch_mean,total_loss_epoch_mean]


def render_novel_view(image_data,c2w,t, H, W, fxfy, ray_params, nerf_model):

    nerf_model.eval()

    #set up
    num_rows = H // ray_params.Win_H
    num_cols = W // ray_params.Win_W

    IMGS = image_data[t] # (N,H,W,3)

    ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(ray_params.NEAR, ray_params.FAR, ray_params.N_SAMPLE, device=device)  # (N_sample,) sample position

    c2w = c2w.to(device)  # (4, 4)

    # split an image to rows when the input image resolution is high
    rendered_img = []
    rendered_depth = []

    for row_i in range(num_rows):

        row_start = row_i * ray_params.Win_H
        row_end  = (row_i+1) * ray_params.Win_H

        rgb_row = []
        depth_row = []

        for col_j in range(num_cols):

            col_start = col_j * ray_params.Win_W
            col_end = (col_j + 1) * ray_params.Win_W

            #crop patch
            IMGS_input = rearrange( IMGS[:,row_start:row_end,col_start:col_end,:] , "b h w c -> b c h w") # (N,H,W,3) -> (N,3,H,W)
            IMGS_input = IMGS_input.to(device) #(N,3,N_select_rows, N_select_cols)

            ray_selected_cam = ray_dir_cam[row_start:row_end,col_start:col_end,:]  # (N_select_rows, N_select_cols, 3)

            render_result = model_render_image(IMGS_input,c2w, ray_selected_cam, t_vals, ray_params,
                                            H, W, fxfy, nerf_model,
                                            perturb_t=False, sigma_noise_std=0.0)

            rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)
            depth_map = render_result['depth_map']  # (N_select_rows, N_select_cols)

            #save
            rgb_row.append(rgb_rendered)
            depth_row.append(depth_map)

        rgb_row = torch.cat(rgb_row,dim=1)
        depth_row = torch.cat(depth_row,dim=1)

        rendered_img.append(rgb_row)
        rendered_depth.append(depth_row)

    # combine rows to an image
    rendered_img = torch.cat(rendered_img, dim=0)  # (H, W, 3)
    rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)

    return rendered_img, rendered_depth


#Training

# Initialise all trainabled parameters
focal_net = LearnFocal(H, W, req_grad=True)
if load_model:
    focal_net.load_state_dict(torch.load(f"{model_weight_dir}/{scene_name}_focal.pt",map_location=device))

focal_net = nn.DataParallel(focal_net,device_ids=device_ids)
focal_net.to(device)

pose_param_net = LearnPose(num_cams=N_IMGS, learn_R=True, learn_t=True)
if load_model:
    pose_param_net.load_state_dict(torch.load(f"{model_weight_dir}/{scene_name}_pose.pt",map_location=device))

pose_param_net = nn.DataParallel(pose_param_net,device_ids=device_ids)
pose_param_net.to(device) 

# Get a tiny NeRF model. Hidden dimension set to 128
nerf_model = UNerf(enc_in_dim = 3*N_IMGS,tpos_in_dim=63,dir_in_dim=27) #TinyNerf(pos_in_dims=63, dir_in_dims=27, D=128).cuda()
if load_model:
    nerf_model.load_state_dict(torch.load(f"{model_weight_dir}/{scene_name}_nerf.pt",map_location=device))

nerf_model = nn.DataParallel(nerf_model,device_ids=device_ids)
nerf_model.to(device)

# Set lr and scheduler: these are just stair-case exponantial decay lr schedulers.
opt_nerf = torch.optim.Adam(nerf_model.parameters(), lr=0.001)
opt_focal = torch.optim.Adam(focal_net.parameters(), lr=0.001)
opt_pose = torch.optim.Adam(pose_param_net.parameters(), lr=0.001)

scheduler_nerf = MultiStepLR(opt_nerf, milestones=list(range(0, 10000, 10)), gamma=0.99)
scheduler_focal = MultiStepLR(opt_focal, milestones=list(range(0, 10000, 100)), gamma=0.9)
scheduler_pose = MultiStepLR(opt_pose, milestones=list(range(0, 10000, 100)), gamma=0.9)

# Set tensorboard writer
writer = SummaryWriter(log_dir=os.path.join('logs', scene_name, str(datetime.datetime.now().strftime('%y%m%d_%H%M%S'))))

# Training
print('Start Training... ')
for epoch_i in tqdm(range(N_EPOCH), desc='Training'):
    
    Tr_loss = train_one_epoch(image_data, H, W, ray_params, opt_nerf, opt_focal,
                              opt_pose, nerf_model, focal_net, pose_param_net)


    #save check point
    torch.save(nerf_model.module.state_dict(),f"{model_weight_dir}/{scene_name}_nerf.pt")
    torch.save(focal_net.module.state_dict(),f"{model_weight_dir}/{scene_name}_focal.pt")
    torch.save(pose_param_net.module.state_dict(),f"{model_weight_dir}/{scene_name}_pose.pt")

    fxfy = focal_net()
    #print('epoch {0:4d} Training PSNR {1:.3f}, estimated fx {2:.1f} fy {3:.1f}'.format(epoch_i, train_psnr, fxfy[0], fxfy[1]))
    print(f"epoch {epoch_i+1}: Training PSNR {Tr_loss[0]}, Training SSIM {Tr_loss[1]}, Total_loss {Tr_loss[2]}, estimated fx {fxfy[0]} fy {fxfy[1]}")

    scheduler_nerf.step()
    scheduler_focal.step()
    scheduler_pose.step()

    learned_c2ws = torch.stack([pose_param_net(i) for i in range(N_IMGS)])  # (N, 4, 4)

    #evaluation
    with torch.no_grad():

        if (epoch_i+1) % EVAL_INTERVAL == 0:

            eval_c2w = torch.eye(4, dtype=torch.float32)  # (4, 4)
            fxfy = focal_net()
            rendered_img, rendered_depth = render_novel_view(image_data,eval_c2w,(epoch_i+1)%TSteps, H, W, fxfy, ray_params, nerf_model)
            imageio.imwrite(os.path.join(f"{os.getcwd()}/nvs_midImg/{scene_name}", scene_name + f"_img{epoch_i+1}.png"),(rendered_img*255).cpu().numpy().astype(np.uint8))
            imageio.imwrite(os.path.join(f"{os.getcwd()}/nvs_midImg/{scene_name}", scene_name + f"_depth{epoch_i+1}.png"),(rendered_depth*200).cpu().numpy().astype(np.uint8))

print('Training Completed !')



#Render Result
# Render novel views from a sprial camera trajectory.
# The spiral trajectory generation function is modified from https://github.com/kwea123/nerf_pl.
#from nerfmm.utils.pose_utils import create_spiral_poses

# Render full images are time consuming, especially on colab so we render a smaller version instead.
import time

resize_ratio = 1
with torch.no_grad():
    optimised_poses = torch.stack([pose_param_net(i) for i in range(N_IMGS)])
    radii = np.percentile(np.abs(optimised_poses.cpu().numpy()[:, :3, 3]), q=75, axis=0)  # (3,)
    spiral_c2ws = create_spiral_poses(radii, focus_depth=3.5, n_poses=TSteps, n_circle=1)
    spiral_c2ws = torch.from_numpy(spiral_c2ws).float()  # (N, 3, 4)

    # change intrinsics according to resize ratio
    fxfy = focal_net()
    novel_fxfy = fxfy / resize_ratio
    novel_H, novel_W = H // resize_ratio, W // resize_ratio

    print('NeRF trained in {0:d} x {1:d} for {2:d} epochs'.format(H, W, N_EPOCH))
    print('Rendering novel views in {0:d} x {1:d}'.format(novel_H, novel_W))

    time_list = [i for i in range(TSteps)]
    novel_img_list, novel_depth_list = [], []

    #record processing time
    curr_time = time.time()

    for i in tqdm(range(spiral_c2ws.shape[0]), desc='novel view rendering'):
        novel_img, novel_depth = render_novel_view(image_data,spiral_c2ws[i],time_list[i],novel_H, novel_W, novel_fxfy,
                                                   ray_params, nerf_model)
        novel_img_list.append(novel_img)
        novel_depth_list.append(novel_depth)

    print('Novel view rendering done. Saving to GIF images...')
    print(f"It takes {time.time()-curr_time} to  complete the whole process")

    novel_img_list = (torch.stack(novel_img_list) * 255).cpu().numpy().astype(np.uint8)
    novel_depth_list = (torch.stack(novel_depth_list) * 200).cpu().numpy().astype(np.uint8)  # depth is always in 0 to 1 in NDC

    #os.makedirs('nvs_results', exist_ok=True)
    imageio.mimwrite(os.path.join(f"{os.getcwd()}/nvs_result", scene_name + '_img.gif'), novel_img_list, fps=30)
    imageio.mimwrite(os.path.join(f"{os.getcwd()}/nvs_result", scene_name + '_depth.gif'), novel_depth_list, fps=30)
    print('GIF images saved.')
