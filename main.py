import numpy as np
from scipy.sparse import coo_matrix, vstack, diags
from scipy.sparse.linalg import cg
import mcubes
import time
import open3d as o3d
import argparse
import os

import numpy as np
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import cg
import mcubes
import time
import open3d as o3d
import argparse
import os

class PoissonSurfaceReconstructor:
    def __init__(self, P, N, nx, ny, nz, padding):
        self.P = P
        self.N = N
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.padding = padding

    def fd_partial_derivative(self, h, direction):
        primary_grid_idx = np.arange(self.nx * self.ny * self.nz).reshape((self.nx, self.ny, self.nz))
        if direction == "x":
            num_staggered_grid = (self.nx - 1) * self.ny * self.nz
            row_idx = np.arange(num_staggered_grid)
            col_idx = primary_grid_idx[1:, ...].flatten()
            offset = self.ny * self.nz
        elif direction == "y":
            num_staggered_grid = self.nx * (self.ny - 1) * self.nz
            row_idx = np.arange(num_staggered_grid)
            col_idx = primary_grid_idx[:, 1:, :].flatten()
            offset = self.nx * self.nz
        elif direction == "z":
            num_staggered_grid = self.nx * self.ny * (self.nz - 1)
            row_idx = np.arange(num_staggered_grid)
            col_idx = primary_grid_idx[:, :, 1:].flatten()
            offset = self.nx * self.ny

        # create a diagonal matrix
        return diags([1/h, -1/h], [offset,0], shape=(num_staggered_grid, nx * ny * nz)).tocoo()

    def fd_grad(self):
        Dx = fd_partial_derivative(nx, ny, nz, hx, "x")
        Dy = fd_partial_derivative(nx, ny, nz, hy, "y")
        Dz = fd_partial_derivative(nx, ny, nz, hz, "z")
        return vstack((Dx, Dy, Dz))

    def trilinear_interpolation_weights(self, corner, hx, hy, hz, direction=None):
        if direction in ('x', 'y', 'z'):
            corner_shift = np.array([0.5 * hx if direction == 'x' else 0,
                                    0.5 * hy if direction == 'y' else 0,
                                    0.5 * hz if direction == 'z' else 0])

        corner += corner_shift
    
        relative_coords = (P - corner) / np.array([hx, hy, hz])  # (N, 3)
        lower_indices = np.floor(relative_coords).astype(int)  # (N, 3)
        upper_indices = lower_indices + 1  # (N, 3)
        t = relative_coords - lower_indices  # (N, 3)

        # Compute trilinear interpolation weights
        weights = np.prod(np.stack([(1 - t), t], axis=-1), axis=-2)  # (N, 2, 2, 2)

        if direction == 'x':
            num_grids = (self.nx - 1) * self.ny * self.nz
            staggered_grid_idx = np.arange(num_grids).reshape((self.nx - 1, self.ny, self.nz))
        elif direction == 'y':
            num_grids = self.nx * (self.ny - 1) * self.nz
            staggered_grid_idx = np.arange(num_grids).reshape((self.nx, self.ny - 1, self.nz))
        elif direction == 'z':
            num_grids = self.nx * self.ny * (self.nz - 1)
            staggered_grid_idx = np.arange(num_grids).reshape((self.nx, self.ny, self.nz - 1))
        else:
            num_grids = self.nx * self.ny * self.nz
            staggered_grid_idx = np.arange(num_grids).reshape((self.nx, self.ny, self.nz))

        indices = np.stack([lower_indices, upper_indices], axis=-1)  # (N, 3, 2)
        col_idx = staggered_grid_idx[tuple(indices.transpose(1, 0, 2))]  # (3, N, 2, 2, 2)

        return coo_matrix((weights.flatten(), (np.repeat(np.arange(P.shape[0]), 8), col_idx.flatten())), shape=(P.shape[0], num_grids))

    def reconstruct(self, save_path):
        bbox_size = np.max(P, 0) - np.min(P, 0)

        hx, hy, hz = bbox_size / np.array([self.nx, self.ny, self.nz])
        bottom_left_front_corner = np.min(P, 0) - padding * np.array([hx, hy, hz])

        self.nx += 2 * padding
        self.ny += 2 * padding
        self.nz += 2 * padding

        G = fd_grad(self.nx, self.ny, self.nz, hx, hy, hz)

        weights_params = (self.nx, self.ny, self.nz, bottom_left_front_corner, P, hx, hy, hz)
        Wx = trilinear_interpolation_weights(*weights_params, direction="x")
        Wy = trilinear_interpolation_weights(*weights_params, direction="y")
        Wz = trilinear_interpolation_weights(*weights_params, direction="z")
        W = trilinear_interpolation_weights(*weights_params)

        vx, vy, vz = Wx.T @ N[:, 0], Wy.T @ N[:, 1], Wz.T @ N[:, 2]
        v = np.concatenate([vx, vy, vz])

        print("Start solving for the characteristic function!")
        tic = time.time()
        g, _ = cg(G.T @ G, G.T @ v, maxiter=2000, tol=1e-5)
        toc = time.time()
        print(f"Linear solver finished! {toc-tic:.2f} sec")

        sigma = np.mean(W @ g)
        g -= sigma

        g_field = g.reshape(self.nx, self.ny, self.nz)
        vertices, triangles = mcubes.marching_cubes(g_field, 0)

        vertices = vertices * np.array([hx, hy, hz]) + bottom_left_front_corner

        mcubes.export_obj(vertices, triangles, save_path)
        print(f"{save_path} saved")


def isFile(filename):
    if os.path.isfile(filename):
        return filename 
    else:
        raise FileNotFoundError(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-path', type=isFile)
    parser.add_argument('-x', type=int, default=64)
    parser.add_argument('-y', type=int, default=64)
    parser.add_argument('-z', type=int, default=64)
    parser.add_argument('-p','--padding', type=int, default=8)
    args = parser.parse_args()

    data_dir = os.path.dirname(args.file_path)
    filename = os.path.basename(args.file_path).split(".")[0]
    save_path = os.path.join(data_dir, f"PSR_nx_{args.x}_ny_{args.y}_nz_{args.z}_"+file_name+".obj")

    pcd = o3d.io.read_point_cloud(args.path)
    # point 
    P = np.asarray(pcd.points)
    # normal
    N = np.asarray(pcd.normals)

    psr = PoissonSurfaceReconstructor(P, N, args.x, args.y, args.z, args.padding)
    psr.reconstruct(save_path)