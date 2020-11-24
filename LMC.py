import torch
import numpy as np
import torch.nn as nn
'''
The main code is LMC
'''
def normalize_keypoints(kpts, image_shape):    
    height, width = image_shape[1],image_shape[0]
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height,one*width, one*height])[None]
    center = size / 2
    return (kpts - center[:, None,None, :]) / 1050

class ACN(nn.Module):
    def __init__(self, inc):
        super(ACN, self).__init__()
        self.conv = nn.Conv2d(inc, 2, 1, 1)
        self.eps = 1e-5

    def forward(self, x):
        # Preprocess: reshape x
        len_shp = len(x.shape)
        if len_shp == 3:
            x = x[..., None]
        assert len(x.shape) == 4, "Illegal x"
        #print('x:',x,x.shape)
        # normalize x
        """
        support 2d attention
        """
        attention = self.conv(x)
        l_logit = attention[:, :1, :, :]
        g_logit = attention[:, -1:, :, :]

        # a_l, a_g: B1WH
        a_l = torch.sigmoid(l_logit)
        a_g = nn.Softmax(2)(g_logit.view(*g_logit.shape[:2], -1)).view_as(g_logit)

        # Merge a_l and a_g
        a = a_l * a_g
        a = a / torch.sum(a, dim=(2, 3), keepdim=True)

        # mean: BC11
        mean = torch.sum(x * a, dim=(2, 3), keepdim=True)
        out = x - mean
        std = torch.sqrt(
            torch.sum(a * out ** 2, dim=(2, 3), keepdim=True) + self.eps)
        out = out / std
        return out

class ARB(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
            out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            ACN(channels),
            nn.GroupNorm(32, channels),
            nn.ReLU(),
            nn.Conv2d(channels, out_channels, kernel_size=1),
            ACN(out_channels),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

def index(xs,kn):
    xs = xs
    b,n,k=xs.shape
    for i in range(b):
        distmat = torch.sum( (xs[i].unsqueeze(1) - xs[i].unsqueeze(0)) ** 2,  dim =2, keepdim=True)
        # Choose K best from N
        idx_sort = torch.argsort(distmat, dim=1)[:, :kn]    # number
        if i==0:
            group_output = idx_sort.unsqueeze(0)
        else:
            idx_sort =idx_sort.unsqueeze(0)
            group_output = torch.cat([group_output,idx_sort],dim=0)
    return group_output.squeeze(-1)   # [8,2500,8]

def groupingMSG(input_feature, input_index):
    input_index = input_index.cuda()

    N = input_index.shape[1]
    k1 = np.array([4,  8])
  
    for i in range(input_feature.size(0)):
        group_i = torch.cat([input_feature[i].transpose(0, 1)[input_index[i, :, :a]].view(1,N,-1) for a in k1], dim=2)
        if i < 1:
            group_output = group_i
        else:
            group_output = torch.cat([group_output, group_i], dim=0)
    return group_output.transpose(1, 2).unsqueeze(-1)

class LMC_network(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels    #channels=128
        self.layer_num = depth
        #-------first conv
        self.conv1 = nn.Conv2d(input_channel,channels//4, kernel_size=1)  #input_channel=6
        self.bn1=nn.BatchNorm2d(channels//4)
        self.in1=nn.InstanceNorm2d(channels//4)
        self.re = nn.ReLU()
        #-------
        self.conv2 = nn.Conv2d(3 * channels, channels, kernel_size=1)

        self.MLP=[]
        self.MLP.append(ARB(channels, channels))
        for _ in range((self.layer_num * 2-1)):
            self.MLP.append(ARB(channels))
        self.MLP = nn.Sequential(*self.MLP)

        self.output = nn.Conv2d(channels, 1, kernel_size=1)



    def forward(self, nn_index ,data, xs):
        x1_1 = self.conv1(data)
        x1_1 = self.re(self.bn1(self.in1(x1_1)))
        grouping_data1 = groupingMSG(x1_1, nn_index)
        grouping_data1 =self.conv2( grouping_data1)
        out = self.MLP(grouping_data1)
        logits = torch.squeeze(torch.squeeze(self.output(out), 3), 1)
        return logits

class LMC(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        depth_each_stage = config.net_depth // (config.iter_num + 1)
        self.side_channel = (config.use_ratio == 2) + (config.use_mutual == 2)
        self.weights_init = LMC_network(config.net_channels, 6 + self.side_channel,depth_each_stage, config.clusters)
        self.image_zise = config.image_zise

    def forward(self, data):
        initial_kpt = data['xs']
        V_index = initial_kpt[:, :, :2] - initial_kpt[ :, :, 2:4]
        nn_index = index(V_index,8)
        data['xs'] = data['xs'].unsqueeze(1)
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        data['xs'] = normalize_keypoints(data['xs'], self.image_zise)
        input = data['xs'].transpose(1, 3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1, 2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)
        res_logits, res_e_hat = [], []
        # that right
        V = input[:,:2,:,:]-input[:,2:4,:,:]
        input=torch.cat([V,input], dim=1)
        logits= self.weights_init(nn_index ,input, data['xs'])
        res_logits.append(logits)
        return res_logits







