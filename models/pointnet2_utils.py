import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np



def timeit(tag,t):
    print(f"{tag} took {time() - t:.4f} seconds")
    return time()


def pc_normalize(pc):
    l=pc.shape[0]
    centroid=np.mean(pc,axis=0)
    m=np.max(np.sqrt(np.sum(pc**2,axis=1)))
    pc=pc/m
    return pc



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance=torch.ones(B,N).to(device)*1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:,i]=farthest
        centroid=xyz[batch_indices,farthest,:].view(B,1,3)
        dist=torch.sum((xyz-centroid)**2,dim=-1)
        distance = torch.min(distance, dist)
        farthest=torch.max(distance,dim=-1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    # Sampling
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    new_points = index_points(points, fps_idx)

    # Grouping
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_points = index_points(points, idx)

    # NOTE: comment out these lines
    # grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    # if points is not None:
    #     grouped_points = index_points(points, idx)
    #     new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    # else:
    #     new_points = grouped_xyz_norm
    # if returnfps:
    #     return new_xyz, new_points, grouped_xyz, fps_idx
    if returnfps:
        return new_xyz, new_points, grouped_xyz, grouped_points
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel=out_channel\
    
        self.group_all=group_all
        self.affine_alpha_first = nn.Parameter(torch.ones([1, 1, 1, in_channel//2]))
        self.affine_beta_first = nn.Parameter(torch.zeros([1, 1, 1, in_channel//2]))

        self.affine_alpha_second = nn.Parameter(torch.ones([1, 1, in_channel//2]))
        self.affine_beta_second = nn.Parameter(torch.zeros([1, 1, in_channel//2]))
    
    def DualNorm(self,grouped_points,new_points,B):
        #  group: [B,C,32,9]  new: [b,c,9]
 
        # PN

        mean = new_points.mean(dim=2)
        std= torch.std((grouped_points-mean).reshape(B,-1),dim=1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        grouped_points=(grouped_points-mean)/(std+1e-6)
        grouped_points = self.affine_alpha_first*grouped_points + self.affine_beta_first

        # RPN
        
        mean=torch.mean(grouped_points,dim=-2).unsqueeze(dim=-2)
        new_points=new_points.unsqueeze(dim=-2)
        std=torch.std((grouped_points-mean).reshape(B,-1),dim=1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
        new_points=(new_points-mean)/(std+1e-6)
        new_points=self.affine_alpha_second*new_points+self.affine_beta_second

        return grouped_points,new_points
    
    def forward(self,xyz,points):

        xyz=xyz.permute(0,2,1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz,new_points=sample_and_group_all(xyz,points)
        else:
            new_xyz,new_points,grouped_xyz,grouped_points=\
                sample_and_group(self.npoint,self.radius,self.nsample,xyz,points,returnfps=True)
        
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C] # NOTE: C+D -> C

        # NOTE: add DualNorm calculation HERE
        B = new_xyz.shape[0]
        grouped_points, new_points = self.DualNorm(grouped_points, new_points, B)

         # new_points = new_points.unsqueeze(dim=-2).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points,new_points.repeat(1,1,grouped_points.size()[2],1)],dim=-1)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C, nsample,npoint] # NOTE: C+D -> C
  
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat
    

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class Attention(nn.Module):
    """
    Lớp self-attention cho đặc trưng điểm trong point cloud.
    Dùng sau mỗi tầng Set Abstraction để tăng hiệu suất mô hình học hình học.
    Input: [B, C, N] (batch, channel, num_points)
    Output: [B, C, N]
    """
    def __init__(self, in_channels, heads=4):
        super(Attention, self).__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dk = in_channels // heads
        assert in_channels % heads == 0, "in_channels phải chia hết cho số heads"
        self.query = nn.Conv1d(in_channels, in_channels, 1)
        self.key = nn.Conv1d(in_channels, in_channels, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.proj = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.shape
        Q = self.query(x).view(B, self.heads, self.dk, N)  # [B, heads, dk, N]
        K = self.key(x).view(B, self.heads, self.dk, N)
        V = self.value(x).view(B, self.heads, self.dk, N)
        attn = torch.einsum('bhdk,bhdk->bhdn', Q, K) / (self.dk ** 0.5)  # [B, heads, N, N]
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhdn,bhdn->bhdk', attn, V)  # [B, heads, dk, N]
        out = out.contiguous().view(B, C, N)
        out = self.proj(out)
        return out + x  # residual
    

class GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear_main = nn.Linear(in_dim, out_dim)
        self.linear_gate = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.linear_main(x) * torch.sigmoid(self.linear_gate(x))
    
class LightHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.conv1 = nn.Conv1d(in_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
        

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(num_classes)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.global_pool(x)
        x = self.dropout1(self.bn1(self.conv1(x)))
        x = self.dropout2(self.bn2(self.conv2(x)))
        x= x.squeeze(dim=-1)
        return x
    
class TransformerHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(in_channels, in_channels)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=in_channels, nhead=num_heads, batch_first=True
        )
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.proj(x.transpose(1, 2))  # [B, N, C]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global token
        return self.classifier(x)


        