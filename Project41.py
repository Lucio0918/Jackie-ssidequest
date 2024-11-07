import numpy as np
import matplotlib.pyplot as plt
import json
from nibabel.affines import voxel_sizes
from volumeViewer import *
from skimage.io import imread
from scipy.ndimage import map_coordinates
from Interactive_PointBasedRegistration import *
from ImageProjection import *

import json
import gdown
import os


# file interactions:
    # volumeViewer.py: imports this file and uses it to display images
    # Interactive_PointBasedRegistration.py: imports this file, uses it for point-based registration
    # ImageProjection.py: imports and uses it to project image from one space to another

# links to google drive file containing Project4.json
project4_file_id = '1V75MLZ9OW6Q3orZnFwDxyOmoaLF6Ngw0'
project4_url = f'https://drive.google.com/uc?id={project4_file_id}'
project4_path = 'Project4.json'

# download the file if it doesn't already exist locally
if not os.path.exists(project4_path):
    print("Downloading Project4.json from Google Drive...")
    gdown.download(project4_url, project4_path, quiet=False)
else:
    print("File already exists locally.")

# load JSON data
with open(project4_path, encoding='utf-8-sig') as f:
    dt = json.load(f)



landmarks = np.array(dt['Proj4landmarks'])

# deformation field data
D=np.array(dt['D'])

# transformation matrix (from CT to T1 space)
CT2T1= np.array(dt['CT2T1'])

# imaging data for CT, CTwarp, and T1
ct=np.array(dt['ct']['data'])
ctwarp= np.array(dt['ctwarp']['data'])
t1= np.array(dt['t1']['data'])

# voxel sizes for CT and T1 images
ctvoxsz=np.array(dt['ct']['voxsz'])
t1voxsz=np.array(dt['t1']['voxsz'])


 # Print statements for metadata + shapes
print(dt['t1']['dim'])
print(dt['t1']['voxsz'])
print(dt['t1']['orient'])

print("CT")
print(dt['ct']['dim'])
print(dt['ct']['voxsz'])

print("CTWARP")
print(dt['ctwarp']['dim'])
print(dt['ctwarp']['voxsz'])

print("Original landmarks:\n", landmarks)
print("Shape of landmarks:", np.shape(landmarks))
print("Shape of D:", np.shape(D))
print("Shape of CT2T1:", np.shape(CT2T1))
print("Shape of CT", np.shape(ct))

#
# # Define transformations as [theta (degrees), translation vector]
# transangles = [[10, -15, 5], [5, 10, 2], [-5, -5, -7]]
# translations = [[2, -1, 4], [-1, 4, 0], [-0.5, -3, -4]]
#
# def create_transformation_matrix(angles, translation):
#     """Create a 4x4 transformation matrix for a rotation and translation in 3D."""
#     # Convert angles to radians
#     theta_x, theta_y, theta_z = np.radians(angles)
#     tx, ty, tz = translation
#
#     # Rotation matrices around x, y, and z axes
#     Rx = np.array([[1, 0, 0],
#                    [0, np.cos(theta_x), -np.sin(theta_x)],
#                    [0, np.sin(theta_x), np.cos(theta_x)]])
#     Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
#                    [0, 1, 0],
#                    [-np.sin(theta_y), 0, np.cos(theta_y)]])
#     Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
#                    [np.sin(theta_z), np.cos(theta_z), 0],
#                    [0, 0, 1]])
#
#     # Combined rotation matrix
#     R = Rz @ Ry @ Rx
#
#     # Create a 4x4 transformation matrix
#     T = np.eye(4)
#     T[:3, :3] = R  # Top-left 3x3 part is the rotation matrix
#     T[:3, 3] = [tx, ty, tz]  # Last column is the translation vector
#
#     return T
#
# # Convert landmarks to homogeneous coordinates (add a fourth coordinate of 1)
# homogeneous_landmarks = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])
#
#
# # Apply each transformation sequentially
# for i, (angles, translation) in enumerate(zip(transangles, translations), start=1):
#
#     T = create_transformation_matrix(angles, translation)
#     # Print the transformation matrix before applying it
#     print(f"\nTransformation matrix T{i} before applying to landmarks:\n", T)
#
#     # Apply the transformation
#     homogeneous_landmarks = (T @ homogeneous_landmarks.T).T
#     print(f"Landmarks after Transformation {i}:\n", homogeneous_landmarks)
#
#     # Plot the transformed landmarks
#     transformed_landmarks = homogeneous_landmarks[:, :3]
#
#
# #Transformation:
# print("First Transformation:", transformed_landmarks[0])
# print("Second Transformation:", transformed_landmarks[1])
# print("Third Transformation:", transformed_landmarks[2])



# #image transformation of CT2T1 and project CT image to T1 space

# projected_image = imageProjection(source = dt['ct']['data'],
#                                   targetdim = dt['t1']['dim'],
#                                   sourcevoxsz = dt['ct']['voxsz'],
#                                   targetvoxsz = dt['t1']['voxsz'],
#                                   T = [CT2T1],
#                                   D = [])


#Using the transformation CT2T1, projected the CT image to T1 space
# vv=volumeViewer()
# vv.setImage(t1 ,dt['t1']['voxsz'])
# vv.display()



#
# #Using the deformation field, projecting the CTwrap image to ct spcae
# projected_ctwrapimage = imageProjection(source = dt['ctwarp']['data'],
#                                    targetdim = dt['ct']['dim'],
#                                    sourcevoxsz = dt['ctwarp']['voxsz'],
#                                    targetvoxsz = dt['ct']['voxsz'],
#                                    T = [],
#                                    D = dt['D'])



#Using the transformation CT2T1, projected the CT image to T1 space
# vv=volumeViewer()
# vv.setImage( projected_image,dt['t1']['voxsz'])
# vv.display()


# #Using the transformation CT2T1, projected the CT image to T1 space
# vv=volumeViewer()
# vv.setImage(projected_ctwrapimage,dt['ct']['voxsz'])
# vv.display()


#ipr = interactive_pointBasedRegistration(ct, ctvoxsz, t1, t1voxsz )

# fudicialdata=ipr.T1to2
# fud1=np.linalg.inv(fudicialdata)
# final=np.dot(fud1, CT2T1)
# print(final)

# interactive point based registration (to align warped CT image with original CT image based on voxel size, done by picking landmarks on each image and adjusting one image til landmarks align...great stuff)
# meaning that ipr2 contains transformation matrices + deformation fields
ipr2 = interactive_pointBasedRegistration(ctwarp, ctvoxsz, ct, ctvoxsz)
plt.show()


 # calculate Mean Absolute Difference between Db and D to measure registration accuracy
mad_deformation_field = np.mean(np.abs(D - ipr.Db))

 # calculate the inverse of the transformation matrix to validate/compare transformations
computed_transformation_inv = np.linalg.inv(ipr.T1to2)

 # compute the product of CT2T1 and the inverse of the computed transformation to evaluate alignment
transformation_comparison = np.dot(computed_transformation_inv, CT2T1)

 # extract the translation component and calculate its error in mm to assess translation error between transformations
translation_error = np.linalg.norm(transformation_comparison[:3, 3])

# check if the translation component error is within 10 mm
print("Transformation Comparison Matrix:\n", transformation_comparison)
print("Translation Component Error (mm):", translation_error)
if translation_error < 10:
    print("Transformation is within the acceptable translation error (< 10 mm).")
else:
    print("Translation error exceeds the acceptable range (> 10 mm).")


print("Mean Absolute Difference", mad_deformation_field)







