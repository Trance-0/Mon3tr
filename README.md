# Mon3tr

Matching and smOoth difussioN for 3d reconsTRuction

This is just a fun project combining [MASt3R](https://github.com/naver/mast3r) and [Smooth-Diffusion](https://github.com/SHI-Labs/Smooth-Diffusion). To test if we can use 2D image generation model to improve 3D reconstruction.

## Key Ideas

From limited sample 2D images, generate a 3D scene that matches the 2D images.

Then for each perspective with information loss, generate a 2D image from the incomplete 3D scene with masking on the missing parts.

Then we ask the smooth-diffusion to complete the 2D image.

Then we use the completed 2D image to reconstruct the 3D scene to obtain the full 3D scene from limited construction.

## Key functions checkpoints

- [x] 3D scene generation from 2D images
- [x] 2D image generation from 3D scene
- [ ] 3D scene reconstruction from incomplete 2D images

- [ ] Build a simple UI to show the process of the key functions



