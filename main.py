import matplotlib.pyplot as plt
from pyramids import *

kernel = smooth_gaussian_kernel(0.6)
kernel = classical_gaussian_kernel(5, 2)
levels = 4
factor = 2
image = plt.imread('/home/rblin/Documents/New_illustrations_ACCV/PARAM_POLAR/I/0001611_I0.png')

plt.imshow(image, cmap='gray')
plt.show()

gauss_pyr = gaussian_pyramid(image, kernel, levels, factor)

lap_pyr = laplacian_pyramid(image, kernel, levels, factor)

i = 1
for p in gauss_pyr:
    plt.subplot(1,len(gauss_pyr),i)
    plt.imshow(p, cmap='gray')
    i += 1
plt.show()

i = 1
for p in lap_pyr:
    plt.subplot(1,len(lap_pyr),i)
    plt.imshow(p, cmap='gray')
    i += 1
plt.show()