# Mon3tr

Matching and smOoth difussioN for 3d reconsTRuction

This is an experimental project as Final project of Spring 2025 CSE 559A Computer Vision course at Washington University in St. Louis. It combines [MASt3R](https://github.com/naver/mast3r) and [Smooth-Diffusion](https://github.com/SHI-Labs/Smooth-Diffusion). To test if we can use 2D image generation model to improve 3D reconstruction.

It has nothing to do with Arknights (perhaps).

![Image credits from [Melanbread](https://www.instagram.com/p/DINiQe8RfgO/)](./assets/Mon3tr.png)

Image credits from [Melanbread](https://www.instagram.com/p/DINiQe8RfgO/)

## Team member

Zheyuan Wu <me@trance-0.com>

## Key Ideas

From limited sample 2D images, generate a 3D scene that matches the 2D images.

Then for each perspective with information loss, generate a 2D image from the incomplete 3D scene with masking on the missing parts.

Then we ask the smooth-diffusion to complete the 2D image.

Then we use the completed 2D image to reconstruct the 3D scene to obtain the full 3D scene from limited construction.

## Key functions checkpoints

- [x] 3D scene generation from 2D images
  - [x] 2D image pair generation with masks
- [x] 2D image generation from 3D scene
  - [x] In-painting 2D image with smooth-diffusion
- [ ] 3D scene reconstruction from incomplete 2D images

- [ ] Build a simple UI to show the process of the key functions
  - [x] 3D scene generation from 2D images
  - [x] 2D inpainting with smooth-diffusion



