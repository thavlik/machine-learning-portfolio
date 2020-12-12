import os
import numpy as np
import torch
import torch.utils.data as data
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.transforms import random_rotation, Translate, Rotate, Transform3d
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
    HardFlatShader,
    TexturesUV,
    TexturesVertex
)


class ToyNeuralGraphicsDataset(data.Dataset):
    def __init__(self,
                 dir: str,
                 rasterization_settings: dict,
                 znear: float = 1.0,
                 zfar: float = 1000.0,
                 scale_min: float = 0.5,
                 scale_max: float = 2.0,
                 device: str = 'cuda'):
        super(ToyNeuralGraphicsDataset, self).__init__()
        device = torch.device(device)
        self.device = device
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.scale_range = scale_max - scale_min
        objs = [os.path.join(dir, f)
                for f in os.listdir(dir)
                if f.endswith('.obj')]
        self.meshes = load_objs_as_meshes(objs, device=device)
        R, T = look_at_view_transform(0, 0, 0)
        self.cameras = FoVPerspectiveCameras(R=R,
                                             T=T,
                                             znear=znear,
                                             zfar=zfar,
                                             device=device)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=RasterizationSettings(
                    **rasterization_settings),
            ),
            shader=HardFlatShader(
                device=device,
                cameras=self.cameras,
            )
        )

    def get_random_transform(self):
        scale = (torch.rand(1).squeeze() *
                 self.scale_range + self.scale_min).item()

        rot = random_rotation()

        x, y, d = torch.rand(3)
        x = x * 2.0 - 1.0
        y = y * 2.0 - 1.0
        trans = torch.Tensor([x, y, d])
        trans = self.cameras.unproject_points(trans.unsqueeze(0).to(self.device),
                                              world_coordinates=False,
                                              scaled_depth_input=True)[0].cpu()
        return scale, rot, trans

    def __getitem__(self, index):
        index %= len(self.meshes)
        scale, rot, trans = self.get_random_transform()
        transform = Transform3d() \
            .scale(scale) \
            .compose(Rotate(rot)) \
            .translate(*trans) \
            .get_matrix() \
            .squeeze()
        mesh = self.meshes[index].scale_verts(scale)
        pixels = self.renderer(mesh,
                               R=rot.unsqueeze(0).to(self.device),
                               T=trans.unsqueeze(0).to(self.device))
        pixels = pixels[0, ..., :3].transpose(0, -1)
        return (pixels, [transform.to(self.device)])

    def __len__(self):
        return len(self.meshes) * 1024


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = ToyNeuralGraphicsDataset('data/',
                                  rasterization_settings=dict(image_size=256,
                                                              blur_radius=0.0,
                                                              faces_per_pixel=1))
    image, labels = ds[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(image.cpu().numpy())
    plt.grid("off")
    plt.axis("off")
    plt.show()
    plt.close('all')
