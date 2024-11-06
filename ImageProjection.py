import numpy as np
import matplotlib.pyplot as plt
import json
from nibabel.affines import voxel_sizes
from volumeViewer import *
from skimage.io import imread
from scipy.ndimage import map_coordinates


def imageProjection(source, targetdim, sourcevoxsz=[], targetvoxsz=[], T=[], D=[]):
    """
    Project a 2D or 3D image array onto a target grid defined by targetdim.

    Parameters:
    - source (np.ndarray): The 2D or 3D source image volume.
    - targetdim (list/np.ndarray): Dimensions of the target grid (e.g., [x, y, z]).
    - sourcevoxsz (list/np.ndarray): Voxel size of the source grid in mm.
    - targetvoxsz (list/np.ndarray): Voxel size of the target grid in mm.
    - T (np.ndarray): 4x4 transformation matrix in mm units (optional).
    - D (np.ndarray): 3 x r x c x d deformation field in voxel units (optional).

    Returns:
    - np.ndarray: Projected source image in the target space.
    """
    #Make an empty array for results
    ans = np.empty((targetdim[0], targetdim[1], targetdim[2]))

    print(np.shape(source))
    print(targetdim)

    if len(T) > 0:
        for i in range(targetdim[0]):
            for j in range(targetdim[1]):
                for k in range(targetdim[2]):
                    target_point = np.array([i * targetvoxsz[0], #put it in xyz coordiantes
                                             j * targetvoxsz[1], #Homogenous coordinate for
                                             k * targetvoxsz[2], #transformation
                                             1])

                    vector_t_p = target_point.reshape(4, 1) #reshape to a colum vector
                    projected_t_p = np.dot(np.linalg.inv(T), vector_t_p) #apply the inverse transformation

                    projected_t_p = [projected_t_p[0, 0],#to map to target voxel back to source
                                     projected_t_p[1, 0],
                                     projected_t_p[2, 0]] # converts from homegous back to 3D
                    #check if poitns are out of bounds and if so apply 0's
                    if projected_t_p[0] < 0 or projected_t_p[1] < 0 or projected_t_p[2] < 0:
                        ans[i][j][k] = 0
                    elif projected_t_p[0] > 322 or projected_t_p[1] > 322 or projected_t_p[2] > 105:
                        ans[i][j][k] = 0
                    else:
                        source_voxel = [int(projected_t_p[0] / sourcevoxsz[0]),    #map the projected coordinate back to
                                        int(projected_t_p[1] / sourcevoxsz[1]), #source coordinates
                                        int(projected_t_p[2] / sourcevoxsz[2])]
                        ans[i][j][k] = source[source_voxel[0]][source_voxel[1]][source_voxel[2]] # assign the corresponding intensity
                        print([i, j, k])
                        #source image back to the target


        return ans

    elif len(D)>0:
        for i in range(targetdim[0]):
            for j in range(targetdim[1]):
                for k in range(targetdim[2]): # Compute the projected source position by subtracting deformation from the target position

                    projected_t_p = np.array([i * targetvoxsz[0] - D[0][i][j][k],
                                          j * targetvoxsz[1] - D[1][i][j][k],
                                          k * targetvoxsz[2] - D[2][i][j][k]])


                    #check if points are out of bounds and if so place them as 0's

                    if projected_t_p[0] < 0 or projected_t_p[1] < 0 or projected_t_p[2] < 0:
                        ans[i][j][k] = 0
                    elif projected_t_p[0] > targetdim[0]*targetvoxsz[0] or projected_t_p[1] > targetdim[1]*targetvoxsz[1] or projected_t_p[2] > targetdim[2]*targetvoxsz[2]:
                        ans[i][j][k] = 0
                    else:
                        #maps the projected points back the source image
                        source_voxel = [int(projected_t_p[0] / sourcevoxsz[0]),
                                        int(projected_t_p[1] / sourcevoxsz[1]),
                                        int(projected_t_p[2] / sourcevoxsz[2])]
                        #assign intensity from source to target
                        ans[i][j][k] = source[source_voxel[0]][source_voxel[1]][source_voxel[2]]
                        print([i, j, k])

        return ans

