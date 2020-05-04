from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg
from random import uniform



"""
Shows 15 images in the given dataset
"""
def show_images_in_dataset(dataset, cmap=plt.cm.bone):
  fig = plt.figure(figsize=(10,5))
  for i in range(15):
    ax = fig.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    ax.imshow(dataset[i], cmap=cmap)



def pca_using_svd(data, num_components):
  # calculate mean
  mean_face = np.mean(data, axis=0)

  # center data
  data = data - mean_face

  # SVD on centered data
  U, S, V = np.linalg.svd(data)

  reduced_u = U[:,:num_components]
  reduced_s = S[:num_components]
  reduced_v = V[:num_components]

  return data, reduced_v, reduced_s, mean_face



def reconstruct_face(face_to_reconstruct, mean_face, eigenvectors):
  # vectorize image
  face_vector = face_to_reconstruct.reshape(-1)
 # subtract mean vector
  face_vector = face_vector - mean_face.reshape(-1)
 # project onto principal components, which means dot product of face to reconstruct with each eigenvector
  weights = np.dot(face_vector, eigenvectors.T)
  # assemble face vector by summing average face with weighted components
  new_vec = mean_face.reshape(-1)
  # each weight * each component + average face
  new_vec = new_vec + np.dot(weights, eigenvectors)
  # reshape vector into facial image
  reconstructed_face = np.reshape(new_vec, (face_to_reconstruct.shape))
  
  return reconstructed_face

