import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, :3]
        rotated_data[k, :, :3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point clouds
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high, size=(batch_data.shape[0], 1, 1))
    batch_data *= scales
    return batch_data

class PointCloudTransform:
    def __call__(self, point_cloud):
        if np.random.random() > 0.5:
            point_cloud = rotate_point_cloud(point_cloud)
        point_cloud = jitter_point_cloud(point_cloud)
        point_cloud = random_scale_point_cloud(point_cloud)
        return point_cloud

# Contrastive Loss Function
def contrastive_loss(features, labels, temperature=0.07):
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()
    
    anchor_dot_contrast = similarity_matrix / temperature
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    exp_logits = torch.exp(logits) * (1 - mask)
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    
    loss = -mean_log_prob_pos
    loss = loss.mean()
    
    return loss
