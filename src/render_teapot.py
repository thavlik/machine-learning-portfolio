import os
import sys
import matplotlib.pyplot as plt
import torch
import pytorch3d
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.transforms import random_rotations, Translate, Rotate, Transform3d
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

device = torch.device("cuda")

mesh = load_objs_as_meshes(['data/cow.obj'], device=device)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
# the difference between naive and coarse-to-fine rasterization.
raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the
# -z direction.
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

n = 3

zfar = 100.0

for i in range(n):
    t = i / n
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
    R, T = look_at_view_transform(0, 0, 0)
    cameras = FoVPerspectiveCameras(device=device,
                                    R=R,
                                    T=T,
                                    zfar=zfar)
    smin = 0.1
    smax = 2.0
    srange = smax - smin
    scale = (torch.rand(1).squeeze() * srange + smin).item()

    # Generate a random NDC coordinate https://pytorch3d.org/docs/cameras
    x, y, d = torch.rand(3)
    x = x * 2.0 - 1.0
    y = y * 2.0 - 1.0
    trans = torch.Tensor([x, y, d]).to(device)
    trans = cameras.unproject_points(trans.unsqueeze(0),
                                     world_coordinates=False,
                                     scaled_depth_input=True)[0]
    rot = random_rotations(1)[0].to(device)
    #transform = Transform3d() \
    #    .scale(scale) \
    #    .compose(Rotate(rot)) \
    #    .translate(*trans)

    # TODO: transform mesh
    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
        )
    )
    images = renderer(mesh.scale_verts(scale),
                      R=rot.unsqueeze(0),
                      T=trans.unsqueeze(0))
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.grid("off")
    plt.axis("off")
    plt.show()
    plt.close('all')
