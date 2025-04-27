# rewrite the pip line to combine the mast3r and smooth-diffusion

import copy
import functools
import sys
import os
from pathlib import Path


ROOT = Path(__file__).parent.parent.resolve()           # folder that holds this script
sys.path.extend([
    str(ROOT / "mast3r" ),             # mast3r package
    str(ROOT / "mast3r" / "dust3r"),             # the sub-package inside mast3r
    str(ROOT / "smooth-diffusion"),              # smooth-diffusion package
])

import numpy as np
import trimesh
import gradio
from PIL import Image, ImageOps
from io import BytesIO
import torch

from scipy.spatial.transform import Rotation
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.demo import SparseGAState
from mast3r.retrieval.processor import Retriever

# add scene cam from dust3r
from dust3r.viz import add_scene_cam, cat_meshes, pts3d_to_trimesh
import tempfile

# main demo rewritten functions
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.demo import set_scenegraph_options
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from mast3r.model import AsymmetricMASt3R
from mast3r.demo import get_args_parser as mast3r_get_args_parser
import gradio as gr
from contextlib import nullcontext

from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, delta=0.5, max_inter_samples=10, silent=False) -> tuple[str, list[Image.Image]]:
    """
    convert the scene output to a glb file and a list of rendered images

    parameters:
        outfile: str, the output file name
        imgs: list[Image.Image], the images
        pts3d: list[np.ndarray], the 3D points
        mask: list[np.ndarray], the mask
        focals: list[float], the focal lengths
        cams2world: list[np.ndarray], the camera poses
        cam_size: float, the camera size
        cam_color: list[tuple[int, int, int]], the camera color
        as_pointcloud: bool, if True, return the pointcloud
        transparent_cams: bool, if True, transparent the cameras
        silent: bool, if True, no print

    Returns:
        outfile: str, the output file name
        rendered_imgs: list[Image.Image], the rendered images
    """         
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    if not transparent_cams:
        # add each camera
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(scene, pose_c2w, camera_edge_color,
                        None if transparent_cams else imgs[i], focals[i],
                        imsize=imgs[i].shape[1::-1], screen_width=cam_size)
            
    # ------------------------------------------------------------------
    # 1. Build intermediate camera poses
    # ------------------------------------------------------------------
    inter: list[tuple[int, np.ndarray]] = []          # (k, pose)

    # helper -----------------------------------
    def slerp_single(Ra: Rotation, Rb: Rotation, t: float) -> Rotation:
        """
        Interpolates between Ra (t=0) and Rb (t=1) with geodesic parameter t.
        Works on any SciPy version (no Rotation.slerp dependency).
        """
        # relative rotation that takes a → b
        R_rel  = Rb * Ra.inv()
        # minimal‐axis representation
        rotvec = R_rel.as_rotvec() * t
        # apply partial relative rotation to Ra
        return Rotation.from_rotvec(rotvec) * Ra
    
    if 0 < delta <= 1 and max_inter_samples > 0:
        base_frac = delta / 3.0
        for A, B in zip(cams2world[:-1], cams2world[1:]):
            Ra, Rb = map(Rotation.from_matrix, (A[:3, :3], B[:3, :3]))
            ta, tb = A[:3, 3], B[:3, 3]
            dir_t  = tb - ta

            for k in range(1, max_inter_samples + 1):
                f = k * base_frac / max_inter_samples
                for frac in (f, 1.0 - f):             # two-sided sampling
                    R_interp = slerp_single(Ra, Rb, frac).as_matrix()
                    t_interp = ta + frac * dir_t

                    pose              = np.eye(4)
                    pose[:3, :3]      = R_interp
                    pose[:3,  3]      = t_interp
                    inter.append((k, pose))

    # order: largest k first
    inter.sort(key=lambda kp: -kp[0])
    inter = inter[:max_inter_samples] if max_inter_samples else inter

    # ------------------------------------------------------------------
    # 2. Render each intermediate pose
    # ------------------------------------------------------------------
    gallery: list[Image.Image] = []
    for k, pose in inter:
        idx   = np.argmin([np.linalg.norm(pose[:3, 3] - c[:3, 3]) for c in cams2world])
        focal = float(focals[idx])
        img   = imgs[idx]
        H, W  = img.shape[:2]

        try:
            scene.camera = trimesh.scene.Camera(resolution=(W, H), focal=(focal, focal))
            scene.camera_transform = pose @ OPENGL
            im_bytes = scene.save_image(resolution=(W, H), visible=False)
            im_rend  = Image.open(BytesIO(im_bytes)).convert("RGB")

            # build binary mask and colorise it
            arr      = np.asarray(im_rend)
            mask_np  = (~np.all(arr == 255, axis=-1)).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np, mode="L")
            mask_rgb = ImageOps.colorize(mask_img, black="black", white="white")

            # side-by-side composite
            combo = Image.new("RGB", (W * 2, H))
            combo.paste(im_rend, (0, 0))
            combo.paste(mask_rgb, (W, 0))

            gallery.append(combo)               # <-- single PIL Image
        except Exception as e:
            if not silent:
                print(f"Render (k={k}) failed: {e}")
    # ------------------------------------------------------------------
    # rest of the function (scene export, etc.) unchanged
    # ------------------------------------------------------------------
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile, gallery

def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0, delta=0.5, max_inter_samples=10) -> tuple[str, list[Image.Image]]:
    """
    extract 3D_model (glb file) from a reconstructed scene

    To fit the project requirement, we extract the camera position and orientation from the scene as return value.

    parameters:
        silent: bool, if True, no print
        scene_state: SparseGAState, the scene state
        min_conf_thr: float, the minimum confidence threshold
        as_pointcloud: bool, if True, return the pointcloud
        mask_sky: bool, if True, mask the sky
        clean_depth: bool, if True, clean the depth

    Returns:
        outfile: str, the output file name
        rendered_imgs: list[Image.Image], the rendered images
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
    outfile, rendered_imgs = _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, delta=delta, max_inter_samples=max_inter_samples, silent=silent)
    return outfile, rendered_imgs

# rewrite get_reconstructed_scene to output gallery
def get_reconstructed_scene(outdir, gradio_delete_cache, model, retrieval_model, device, silent, image_size,
                            current_scene_state, filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr,
                            matching_conf_thr, as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics, delta, max_inter_samples, **kw):
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
    outfile, imgs = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples)

    # Ensure the scene is reloaded correctly for rendering
    scene_file = trimesh.load(outfile, force='scene')

    # render basic view of the scene
    basic_img = scene_file.save_image(resolution=(512, 512), visible=False)
    basic_img = Image.open(BytesIO(basic_img))

    return scene, outfile, imgs, basic_img

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
                    with gradio.Row():
                        delta = gradio.Slider(label="Delta", value=0.5,
                                                          minimum=0., maximum=1., step=0.01,
                                                          info="Delta for intermediate poses")
                        max_inter_samples = gradio.Slider(label="Max Inter Samples", value=10,
                                                          minimum=0, maximum=100, step=1)
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

            with gradio.Row():
                outmodel = gradio.Model3D(height="100%")

                outimage = gradio.Image(label='Output image', height="100%")

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
                          inputs=[
                                    scene,                  # current_scene_state
                                    inputfiles,             # filelist
                                    optim_level,            # optim_level
                                    lr1, niter1, lr2, niter2,
                                    min_conf_thr,           # min_conf_thr
                                    matching_conf_thr,      # matching_conf_thr
                                    as_pointcloud, mask_sky, clean_depth,
                                    transparent_cams, cam_size,
                                    scenegraph_type, winsize, win_cyclic, refid,
                                    TSDF_thresh,            # TSDF_thresh  (only once!)
                                    shared_intrinsics,
                                    delta,                  # new delta slider
                                    max_inter_samples       # new max-samples slider
                                ],
                          outputs=[scene, outmodel, outgallery, outimage])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                            outputs=outmodel)
            TSDF_thresh.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                               outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                                    outputs=outmodel)
            delta.change(fn=model_from_scene_fun,
                         inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                 clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                         outputs=outmodel)
            max_inter_samples.change(fn=model_from_scene_fun,
                                     inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                             clean_depth, transparent_cams, cam_size, TSDF_thresh, delta, max_inter_samples],
                                     outputs=outmodel)
    demo.launch(share=share, server_name=server_name, server_port=server_port)


torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

def get_args_parser():
    parser = mast3r_get_args_parser()
    # change defaults
    parser.prog = 'Mon3tr demo'
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # change default tmp_dir to the current directory
    if args.tmp_dir is None:
        args.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')

    set_print_with_timestamp()

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)

    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mon3tr_gradio_demo') if tmp_dir is None \
            else nullcontext(tmp_dir)
    with get_context(args.tmp_dir) as tmpdirname:
        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        main_demo(cache_path, model, args.retrieval_model, args.device, args.image_size, server_name, args.server_port,
                  silent=args.silent, share=args.share, gradio_delete_cache=args.gradio_delete_cache)
