from jenkspy import JenksNaturalBreaks
import numpy as np

arr = np.array([[ 0.4175,  0.2946, -0.2116, -0.2154],
        [ 0.3566,  0.4245,  0.2321, -0.4188],
        [-0.4813,  0.3807, -0.4328, -0.3908],
        [-0.0824, -0.0291,  0.1397,  0.0277],
        [-0.1358, -0.4866,  0.1868, -0.4537],
        [-0.0392, -0.2796,  0.2498,  0.3536],
        [-0.4195, -0.3004,  0.1421,  0.1625],
        [ 0.0492, -0.3681, -0.3715, -0.0706]])


arr_1 = np.array([[-0.0615, -0.0615, -0.0615, -0.0615],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0277,  0.0277,  0.0277,  0.0277],
        [ 0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0377, -0.0377, -0.0377, -0.0377],
        [-0.0516, -0.0516, -0.0516, -0.0516],
        [ 0.0000,  0.0000,  0.0000,  0.0000]])

arr_score = np.abs(np.multiply(arr, arr_1).flatten())
print(arr_score)

jnb = JenksNaturalBreaks(2)
jnb.fit(arr_score)
print(jnb.labels_)
labels = jnb.labels_
indices = np.where(labels == 1)[0]
indices_ = np.where(labels == 0)[0]

param_data_flat = arr.reshape(-1)
param_grad_flat = arr_1.reshape(-1)

velocity_flat = np.zeros_like(param_data_flat)
momentum = 0.9
scale = 0.9
lr = 0.01

velocity_flat[indices] = momentum * velocity_flat[indices] + scale * param_data_flat[indices] + param_grad_flat[indices]
velocity_flat[indices_] = momentum * velocity_flat[indices_] + scale * param_data_flat[indices_]

# Update parameters
param_data_flat[indices] -= lr * velocity_flat[indices]
param_data_flat[indices_] -= lr * velocity_flat[indices_]

param_data = param_data_flat.reshape(arr.shape)
print(param_data)

goal = np.array([[ 0.4137,  0.2919, -0.2091, -0.2129],
        [ 0.3534,  0.4207,  0.2301, -0.4151],
        [-0.4770,  0.3773, -0.4289, -0.3873],
        [-0.0819, -0.0291,  0.1382,  0.0271],
        [-0.1346, -0.4822,  0.1851, -0.4496],
        [-0.0385, -0.2767,  0.2475,  0.3504],
        [-0.4152, -0.2972,  0.1408,  0.1610],
        [ 0.0487, -0.3648, -0.3682, -0.0700]])

print(np.allclose(param_data, goal, atol=1e-4))