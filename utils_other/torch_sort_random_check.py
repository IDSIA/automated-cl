# from https://stackoverflow.com/questions/60366033/torch-sort-and-argsort-sorting-randomly-in-case-of-same-element
import numpy as np
import torch

# a = np.array(
#         [[ 0., 3.],
#         [ 2., 3.],
#         [ 2., 2.],
#         [10., 2.],
#         [ 0., 2.],
#         [ 6., 2.],
#         [10., 1.],
#         [ 2., 1.],
#         [ 0., 1.],
#         [ 6., 1.],
#         [10., 0.],
#         [12., 0.]]
# )
# 
# numpy_stable_sorted = a[np.argsort(a[:, 0], kind='stable')]

a = torch.tensor(
        [[ 0., 3.],
        [ 2., 3.],
        [ 2., 2.],
        [10., 2.],
        [ 0., 2.],
        [ 6., 2.],
        [10., 1.],
        [ 2., 1.],
        [ 0., 1.],
        [ 6., 1.],
        [10., 0.],
        [12., 0.]]
)

for i in range(10):
    b=a[torch.randperm(a.size()[0])]
    torch_sorted = b[torch.argsort(b[:, 0])].numpy()
    numpy_stable_sorted = b[np.argsort(b[:, 0], kind='stable')]
    print(np.array_equal(torch_sorted, numpy_stable_sorted))
