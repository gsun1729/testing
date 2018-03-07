import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from lib.read_write import *
from skimage import measure
from skimage.draw import ellipsoid
from skimage.data import binary_blobs
from lib.render import *
import sys
# Generate a level set about zero of two identical ellipsoids in 3D
# ellip_base = ellipsoid(6, 10, 16, levelset=True)
# ellip_double = np.concatenate((ellip_base[:-1, ...],
#                                ellip_base[2:, ...]), axis=0)

data = binary_blobs(length=30, blob_size_fraction=0.2, n_dim=3, volume_fraction=0.01, seed=None)

print type(data[0,0,0])
# sys.exit()
# Use marching cubes to obtain the surface mesh of these ellipsoids
verts, faces, normals, values = measure.marching_cubes_lewiner(data, level=None, spacing=(1.0, 1.0, 1.0), gradient_direction='descent', step_size=1, allow_degenerate=True, use_classic=False)
# save_data(verts, "verts", ".\\")
# save_data(faces, "faces", ".\\")
# save_data(normals, "normals", ".\\")
# save_data(values, "values", ".\\")
# save_data(data, "blobs", ".\\")
# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis: a = 6 per ellipsoid")
ax.set_ylabel("y-axis: b = 10")
ax.set_zlabel("z-axis: c = 16")

ax.set_xlim(0, 30)  # a = 6 (times two for 2nd ellipsoid)
ax.set_ylim(0, 30)  # b = 10
ax.set_zlim(0, 30)  # c = 16

plt.tight_layout()
plt.show()
