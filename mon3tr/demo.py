# rewrite the pip line to combine the mast3r and smooth-diffusion

import copy
import functools
import sys
import os
from pathlib import Path

import numpy as np
import trimesh



sys.path.append(os.path.join(Path(__file__).parent, 'mast3r'))
sys.path.append(os.path.join(Path(__file__).parent, 'mast3r','dust3r'))
sys.path.append(os.path.join(Path(__file__).parent, 'smooth-diffusion'))

import gradio
from PIL import Image
from io import BytesIO
import torch

from scipy.spatial.transform import Rotation
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.demo import SparseGAState
from mast3r.retrieval.processor import Retriever

# add scene cam from dust3r
from dust3r.viz import add_scene_cam
import tempfile

# main demo rewritten functions
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.demo import set_scenegraph_options,_convert_scene_output_to_glb
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as pl


OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def render_n_views_around_scene_fixed(scene_file, num_views=1):
    rendered_imgs = []

    # Compute the center of the scene using trimesh
    scene_center = scene_file.centroid
    scene_extent = scene_file.scale
    radius = scene_extent * 1.5  # larger radius for better views

    for i in range(num_views):
        # Calculate angle for counter-clockwise rotation
        theta = 2 * np.pi * (num_views - i) / num_views
        # Place camera in circle around scene, on x-y plane
        cam_position = scene_center + radius * np.array([np.cos(theta), np.sin(theta), 0.2])

        # Compute camera K matrix
        K = np.eye(4)
        K[0, 0] = 1  # Focal length in x
        K[1, 1] = 1  # Focal length in y
        K[0, 2] = cam_position[0]  # Principal point x
        K[1, 2] = cam_position[1]  # Principal point y
        K[2, 2] = 1  # Homogeneous coordinate

        # Compute look-at transform using K matrix
        forward = scene_center - cam_position
        forward /= np.linalg.norm(forward)

        # Simple camera "up" direction
        up = np.array([0, 0, 1])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        rot = np.stack([right, up, -forward], axis=1)
        transform = np.eye(4)
        transform[:3, :3] = rot @ K[:3, :3]
        transform[:3, 3] = cam_position

        # Apply camera transform and render
        scene_file.camera.transform = transform
        try:
            img_bytes = scene_file.save_image(resolution=(512, 512), visible=False)
            img = Image.open(BytesIO(img_bytes))
            rendered_imgs.append(img)
        except Exception as e:
            print(f"View {i} failed: {e}")
    
    return rendered_imgs


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0):
    """
    extract 3D_model (glb file) from a reconstructed scene

    To fit the project requirement, we extract the camera position and orientation from the scene as return value.
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))
    msk = to_numpy([c > min_conf_thr for c in confs])
    return _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent),cams2world,focals,rgbimg

# rewrite get_reconstructed_scene to output gallery
def get_reconstructed_scene(outdir, gradio_delete_cache, model, retrieval_model, device, silent, image_size,
                            current_scene_state, filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr,
                            matching_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)

    sim_matrix = None
    if 'retrieval' in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and not current_scene_state.should_delete and current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and not current_scene_state.should_delete and current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile,cams2world,focals,rgbimg = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    

    # also return the reconstructed scene as a gallery of image from different views of the scene
    print(type(outfile))
    scene_file = trimesh.load(outfile, force='scene')
    imgs=render_n_views_around_scene_fixed(scene_file)
    # for transform,focal,img in zip(cams2world,focals,rgbimg):
    #     screen_width=0.03
    #     if img is not None:
    #         img = np.asarray(img)
    #         H, W, THREE = img.shape
    #         assert THREE == 3
    #         if img.dtype != np.uint8:
    #             img = np.uint8(255*img)
    #     elif focal is not None:
    #         H = W = focal / 1.1
    #     else:
    #         H = W = 1

    #     if isinstance(focal, np.ndarray):
    #         focal = focal[0]

    #     # create fake camera
    #     height = max( screen_width/10, focal * screen_width / H )
    #     width = screen_width * 0.5**0.5
    #     rot45 = np.eye(4)
    #     rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    #     rot45[2, 3] = -height  # set the tip of the cone = optical center
    #     aspect_ratio = np.eye(4)
    #     aspect_ratio[0, 0] = W/H
    #     final_transform = transform @ OPENGL @ aspect_ratio @ rot45
    #     scene_file.camera.transform = final_transform 
    #     if focal > 0:
    #         fx = fy = focal
    #         scene_file.camera.focal = (fx, fy)

    #     # === Render ===
    #     try:
    #         # scene_file.camera.focal = (1/focal,1/focal)
    #         # Debug
    #         print(final_transform,final_transform,scene_file.camera.transform)
    #         print(focal,scene_file.camera.focal)
    #         image_bytes = scene_file.save_image(resolution=(W, H), visible=False)
    #         img_rendered = Image.open(BytesIO(image_bytes))
    #         imgs.append(img_rendered)
    #     except Exception as e:
    #         print(f"Render failed: {e}")

    return scene, outfile, imgs

def main_demo(tmpdirname, model, retrieval_model, device, image_size, server_name, server_port, silent=False,
              share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model,
                                  retrieval_model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    available_scenegraph_type = [("complete: all possible image pairs", "complete"),
                                 ("swin: sliding window", "swin"),
                                 ("logwin: sliding window with long range", "logwin"),
                                 ("oneref: match one image with all", "oneref")]
    if retrieval_model is not None:
        available_scenegraph_type.insert(1, ("retrieval: connect views based on similarity", "retrieval"))

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "Mon3tr Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="Mon3tr Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:  
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">Mon3tr Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                        niter1 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1,
                                               label="Iterations", info="For coarse alignment")
                        lr2 = gradio.Slider(label="Fine LR", value=0.01, minimum=0.005, maximum=0.05, step=0.001)
                        niter2 = gradio.Slider(value=300, minimum=0, maximum=1000, step=1,
                                               label="Iterations", info="For refinement")
                        optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                      value='refine+depth', label="OptLevel",
                                                      info="Optimization level")
                    with gradio.Row():
                        matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=0.,
                                                          minimum=0., maximum=30., step=0.1,
                                                          info="Before Fallback to Regr3D!")
                        shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                            info="Only optimize one set of intrinsics for all views")
                        scenegraph_type = gradio.Dropdown(available_scenegraph_type,
                                                          value='complete', label="Scenegraph",
                                                          info="Define how to make pairs",
                                                          interactive=True)
                        with gradio.Column(visible=False) as graph_opt:
                            winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                    minimum=1, maximum=1, step=1)
                            win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                            refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                                  minimum=0, maximum=0, step=1, visible=False)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
                TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=False, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=True, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=True, label="Transparent cameras")

            outmodel = gradio.Model3D()

            outgallery = gradio.Gallery(label='Output rendered images', columns=3, height="100%")

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[graph_opt, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[graph_opt, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            TSDF_thresh.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                    outputs=outmodel)
    demo.launch(share=share, server_name=server_name, server_port=server_port)
