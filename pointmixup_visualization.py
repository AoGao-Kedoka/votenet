import os
import sys
import numpy as np
import torch
from emd_ import emd_module
EMD = emd_module.emdModule()
from plyfile import PlyData, PlyElement


def get_vertices(file_path):
    with open(file_path, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        mesh_vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)

        mesh_vertices[:, 0] = plydata['vertex']['x']
        mesh_vertices[:, 1] = plydata['vertex']['y']
        mesh_vertices[:, 2] = plydata['vertex']['z']

        # If the file contains color information
        if 'red' in plydata['vertex'].data.dtype.names:
            mesh_vertices[:, 3] = plydata['vertex']['red']
            mesh_vertices[:, 4] = plydata['vertex']['green']
            mesh_vertices[:, 5] = plydata['vertex']['blue']

        return mesh_vertices

def adjust_point_cloud_size(vertices1, vertices2, multiple=128):
    num_points1 = vertices1.shape[0]
    num_points2 = vertices2.shape[0]

    # Determine the target size, which should be the same and a multiple of `multiple`
    target_size = max(num_points1, num_points2)
    target_size = ((target_size + multiple - 1) // multiple) * multiple

    def pad_or_subsample(vertices, target_size):
        current_size = vertices.shape[0]
        if current_size > target_size:
            # Subsample
            indices = np.random.choice(current_size, target_size, replace=False)
            return vertices[indices, :]
        elif current_size < target_size:
            # Pad with zeros
            padding = np.zeros((target_size - current_size, vertices.shape[1]), dtype=vertices.dtype)
            return np.vstack((vertices, padding))
        else:
            return vertices

    vertices1 = pad_or_subsample(vertices1, target_size)
    vertices2 = pad_or_subsample(vertices2, target_size)

    return vertices1, vertices2

def mixup_augmentation(xyz, xyz_minor, mix_rate=0.2):
    B, N, D = xyz.size()
    mix_rate_expand_xyz = mix_rate * torch.ones((B, N, 1), device=xyz.device)
    _, ass = EMD(xyz, xyz_minor, 0.005, 300)
    ass = ass.long()
    xyz_minor_new = torch.zeros_like(xyz)
    for i in range(B):
        xyz_minor_new[i] = xyz_minor[i][ass[i]]
    xyz = xyz * (1 - mix_rate_expand_xyz) + xyz_minor_new * mix_rate_expand_xyz
    return xyz

def save_vertices(vertices, output_file):
    num_verts = vertices.shape[0]
    vertices_tuple = [(vertices[i, 0], vertices[i, 1], vertices[i, 2],
                       vertices[i, 3], vertices[i, 4], vertices[i, 5]) for i in range(num_verts)]
    
    vertex_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertex_data = np.array(vertices_tuple, dtype=vertex_dtype)

    ply_element = PlyElement.describe(vertex_data, 'vertex')
    PlyData([ply_element]).write(output_file)

if __name__=='__main__':
    base_dir = 'scannet/scans_val_transformed'
    output_dir = 'scannet/scans_mixup'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    original_file = os.path.join(base_dir, 'scene0011_00_vh_clean_2.ply')
    interpolate_file = os.path.join(base_dir, 'scene0015_00_vh_clean_2.ply')

    vertices1 = get_vertices(original_file)
    vertices2 = get_vertices(interpolate_file)
    vertices1, vertices2 = adjust_point_cloud_size(vertices1, vertices2, multiple=128)

    vertices1_tensor = torch.tensor(vertices1[:, :3], dtype=torch.float32).unsqueeze(0)  # B, N, D
    vertices2_tensor = torch.tensor(vertices2[:, :3], dtype=torch.float32).unsqueeze(0)  # B, N, D

    for i in range(10):
        mixup_rate = i * 0.1
        output_file = os.path.join(output_dir, f'{mixup_rate}_scene_mixup.ply')
        mixed_vertices = mixup_augmentation(vertices1_tensor, vertices2_tensor, mix_rate= mixup_rate)

        mixed_vertices_np = mixed_vertices.squeeze(0).numpy()
        mixed_vertices_np = np.concatenate((mixed_vertices_np, vertices1[:, 3:]), axis=1)

        save_vertices(mixed_vertices_np, output_file)