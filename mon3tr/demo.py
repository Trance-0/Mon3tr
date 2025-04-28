"""
Mon3tr: MASt3R × Smooth‑Diffusion Demo (fully‑wired)
————————————————————————————————————————————————
• Three‑tab UI (MASt3R, Interpolation, Smooth‑Diffusion)
• Masks produced by MASt3R → piped into Stable‑Diffusion in‑painting
• Working end‑to‑end; ready to launch with `python mon3tr_smooth_diffusion_demo.py`

NOTE  ▸ running the reconstruction requires a CUDA‑capable GPU and the MASt3R
          model weights available (defaults to Naver’s HF repo). The in‑painting
          part needs ~7 GB vRAM with LoRA.
"""

from __future__ import annotations

# ————————————————————————————————————————————————————————————————————
#  Imports & path setup
# ————————————————————————————————————————————————————————————————————
from collections import OrderedDict
from contextlib import nullcontext
import functools, sys, os, os.path as osp, tempfile, copy
from pathlib import Path
from typing import List

ROOT = Path(__file__).parent.parent.resolve()

# local sources --------------------------------------------------------------
sys.path.extend(
    [
        str(ROOT),
        str(ROOT / "mast3r"),
        str(ROOT / "mast3r" / "dust3r"),
        str(ROOT / "smooth-diffusion"),
    ]
)

print(f"DEBUG: sys.path: {sys.path}")

# std / ext ------------------------------------------------------------------
import numpy as np
import torch, trimesh, gradio as gr, matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO
from scipy.spatial.transform import Rotation

plt.ion()

# mast3r ---------------------------------------------------------------------
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.demo import SparseGAState
from mast3r.retrieval.processor import Retriever
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.model import AsymmetricMASt3R
from dust3r.utils.device import to_numpy
from dust3r.utils.image import load_images
from dust3r.viz import add_scene_cam, cat_meshes, pts3d_to_trimesh, CAM_COLORS

# smooth‑diffusion -----------------------------------------------------------
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
)
from huggingface_hub import snapshot_download

# mon3tr ---------------------------------------------------------------------
from mon3tr.mast3r_test import get_reconstructed_scene

# ————————————————————————————————————————————————————————————————————
#  Global constants & choices
# ————————————————————————————————————————————————————————————————————
OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

HERE = ROOT
HF_CACHE = HERE / "smooth-diffusion" / "assets" / "models"
HF_CACHE.mkdir(parents=True, exist_ok=True)
LOW_VRAM = True
VERSION = "Mon3tr: MASt3R × Smooth‑Diffusion Demo v2.0"

choices: dict[str, OrderedDict[str, str]] = OrderedDict()
choices["diffuser"] = OrderedDict(
    [
        ("SD‑v1‑5", "runwayml/stable-diffusion-v1-5"),
        ("OJ‑v4", "prompthero/openjourney-v4"),
        ("RR‑v2", "SG161222/Realistic_Vision_V2.0"),
    ]
)
choices["lora"] = OrderedDict(
    [
        ("empty", ""),
        ("Smooth‑LoRA‑v1", str(HF_CACHE / "smooth_lora.safetensors")),
    ]
)
choices["scheduler"] = OrderedDict([("DDIM", DDIMScheduler)])

default = dict(
    diffuser="RR‑v2",
    scheduler="DDIM",
    lora="Smooth‑LoRA‑v1",
    step=20,
    cfg_scale=7.5,
    strength=0.75,
    blur_radius=10.0,
)

# ————————————————————————————————————————————————————————————————————
#  Helper utilities
# ————————————————————————————————————————————————————————————————————


def _ensure_local(repo_id: str) -> str:
    path = (HF_CACHE / repo_id.replace("/", "--")).as_posix()
    if not osp.isdir(path):
        snapshot_download(repo_id, local_dir=path, resume_download=True)
    return path


def regulate(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((round(w / 64) * 64, round(h / 64) * 64), Image.BILINEAR)


def to_pil(o, mode="RGB") -> Image.Image:
    if isinstance(o, tuple): # for gradio.gallery
        o = o[0]
    if isinstance(o, Image.Image):
        img = o
    elif isinstance(o, np.ndarray):
        img = Image.fromarray(o)
    elif isinstance(o, str):
        img = Image.open(o)
    else:
        raise TypeError(type(o))
    return img.convert(mode)


def blur_mask(msk: Image.Image, r: float) -> Image.Image:
    return msk.filter(ImageFilter.GaussianBlur(radius=r))

def merge_list_states(inpaint_state, original_images_state):
    files = (inpaint_state or [])
    if not isinstance(files, list):         # e.g. a single PIL.Image
        files = [files]

    other = (original_images_state or [])
    if not isinstance(other, list):
        other = [other]

    files += other
    return files
# ————————————————————————————————————————————————————————————————————
#  Smooth‑diffusion wrapper
# ————————————————————————————————————————————————————————————————————
class SmoothWrapper:
    def __init__(self, fp16: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if fp16 else torch.float32
        self.pipe_txt = self.pipe_inp = None
        self.tag_diffuser = self.tag_lora = self.tag_scheduler = None
        self.load_all(default["diffuser"], default["lora"], default["scheduler"])

    # — Loading —
    def load_all(self, d: str, l: str, s: str):
        self._load_diffuser_lora(d, l)
        self._load_scheduler(s)

    def _load_diffuser_lora(self, d: str, l: str):
        if (d, l) == (self.tag_diffuser, self.tag_lora):
            return
        model_path = _ensure_local(choices["diffuser"][d])
        self.pipe_txt = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            safety_checker=None,
            use_safetensors=None,
        ).to(self.device)
        if LOW_VRAM:
            self.pipe_txt.enable_attention_slicing()
            self.pipe_txt.enable_vae_tiling()
        if l != "empty":
            lpath = choices["lora"][l]
            if lpath.endswith((".safetensors", ".bin")):
                self.pipe_txt.unet.load_attn_procs(
                    osp.dirname(lpath), weight_name=osp.basename(lpath)
                )
            else:
                self.pipe_txt.unet.load_attn_procs(lpath)
            self.pipe_txt.unet.to(self.device)
        self.tag_diffuser, self.tag_lora = d, l
        self.pipe_inp = None  # invalidate

    def _load_scheduler(self, s: str):
        if self.tag_scheduler == s and self.pipe_txt is not None:
            return
        sched_cls = choices["scheduler"][s]
        self.pipe_txt.scheduler = sched_cls.from_config(self.pipe_txt.scheduler.config)
        if self.pipe_inp is not None:
            self.pipe_inp.scheduler = self.pipe_txt.scheduler
        self.tag_scheduler = s

    def _get_inpaint(self):
        if self.pipe_inp is not None:
            return self.pipe_inp
        base = choices["diffuser"][self.tag_diffuser]
        self.pipe_inp = StableDiffusionInpaintPipeline.from_pretrained(
            base,
            torch_dtype=self.pipe_txt.unet.dtype,
            low_cpu_mem_usage=True,
            safety_checker=None,
            use_safetensors=None,
        ).to(self.device)
        self.pipe_inp.unet = self.pipe_txt.unet
        self.pipe_inp.vae = self.pipe_txt.vae
        self.pipe_inp.scheduler = self.pipe_txt.scheduler
        if self.tag_lora != "empty":
            lpath = choices["lora"][self.tag_lora]
            if lpath.endswith((".safetensors", ".bin")):
                self.pipe_inp.unet.load_attn_procs(
                    osp.dirname(lpath), weight_name=osp.basename(lpath)
                )
            else:
                self.pipe_inp.unet.load_attn_procs(lpath)
            self.pipe_inp.unet.to(self.device)
        return self.pipe_inp

    # — In‑paint —
    @torch.no_grad()
    def run(
        self,
        imgs: List[Image.Image],
        masks: List[Image.Image],
        prompt: str,
        cfg: float,
        strength: float,
        steps: int,
        radius: float,
    ):
        if len(imgs) != len(masks):
            raise ValueError("image/mask count mismatch")
        pipe = self._get_inpaint()
        blurred, outputs = [], []
        for im, msk in zip(imgs, masks):
            im = regulate(to_pil(im))
            msk = regulate(to_pil(msk, "L"))
            msk_b = blur_mask(msk, radius)
            blurred.append(msk_b)
            out = pipe(
                prompt,
                image=im,
                mask_image=msk_b,
                guidance_scale=cfg,
                strength=strength,
                num_inference_steps=steps,
            ).images[0]
            outputs.append(out)
        return blurred, outputs, outputs


# ————————————————————————————————————————————————————————————————————
#  MASt3R helpers (rendering + mask extraction)
# ————————————————————————————————————————————————————————————————————
def _convert_scene_output_to_glb(
    outfile,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    delta=0.5,
    max_inter_samples=10,
    silent=False,
) -> tuple[str, list[Image.Image]]:
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
            add_scene_cam(
                scene,
                pose_c2w,
                camera_edge_color,
                None if transparent_cams else imgs[i],
                focals[i],
                imsize=imgs[i].shape[1::-1],
                screen_width=cam_size,
            )

    # ------------------------------------------------------------------
    # 1. Build intermediate camera poses
    # ------------------------------------------------------------------
    inter: list[tuple[int, np.ndarray]] = []  # (k, pose)

    # helper -----------------------------------
    def slerp_single(Ra: Rotation, Rb: Rotation, t: float) -> Rotation:
        """
        Interpolates between Ra (t=0) and Rb (t=1) with geodesic parameter t.
        Works on any SciPy version (no Rotation.slerp dependency).
        """
        # relative rotation that takes a → b
        R_rel = Rb * Ra.inv()
        # minimal‐axis representation
        rotvec = R_rel.as_rotvec() * t
        # apply partial relative rotation to Ra
        return Rotation.from_rotvec(rotvec) * Ra

    if 0 < delta <= 1 and max_inter_samples > 0:
        base_frac = delta / 3.0
        for A, B in zip(cams2world[:-1], cams2world[1:]):
            Ra, Rb = map(Rotation.from_matrix, (A[:3, :3], B[:3, :3]))
            ta, tb = A[:3, 3], B[:3, 3]
            dir_t = tb - ta

            for k in range(1, max_inter_samples + 1):
                f = k * base_frac / max_inter_samples
                for frac in (f, 1.0 - f):  # two-sided sampling
                    R_interp = slerp_single(Ra, Rb, frac).as_matrix()
                    t_interp = ta + frac * dir_t

                    pose = np.eye(4)
                    pose[:3, :3] = R_interp
                    pose[:3, 3] = t_interp
                    inter.append((k, pose))

    # order: largest k first
    inter.sort(key=lambda kp: -kp[0])
    inter = inter[:max_inter_samples] if max_inter_samples else inter

    # ------------------------------------------------------------------
    # 2. Render each intermediate pose
    # ------------------------------------------------------------------
    gallery: list[Image.Image] = []
    renders: list[Image.Image] = []
    masks: list[Image.Image] = []
    for k, pose in inter:
        idx = np.argmin([np.linalg.norm(pose[:3, 3] - c[:3, 3]) for c in cams2world])
        focal = float(focals[idx])
        img = imgs[idx]
        H, W = img.shape[:2]

        try:
            scene.camera = trimesh.scene.Camera(resolution=(W, H), focal=(focal, focal))
            scene.camera_transform = pose @ OPENGL
            im_bytes = scene.save_image(resolution=(W, H), visible=False)
            im_rend = Image.open(BytesIO(im_bytes)).convert("RGB")

            # build binary mask and colorise it
            arr = np.asarray(im_rend)
            mask_np = (~np.all(arr == 255, axis=-1)).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np, mode="L")
            # invert the mask
            mask_rgb = ImageOps.colorize(mask_img, black="white", white="black")

            # side-by-side composite
            combo = Image.new("RGB", (W * 2, H))
            combo.paste(im_rend, (0, 0))
            combo.paste(mask_rgb, (W, 0))

            masks.append(mask_rgb)
            renders.append(im_rend)
            gallery.append(combo)
        except Exception as e:
            if not silent:
                print(f"Render (k={k}) failed: {e}")
    # ------------------------------------------------------------------
    # rest of the function (scene export, etc.) unchanged
    # ------------------------------------------------------------------
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile, renders, masks, gallery

def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0, delta=0.5, max_inter_samples=10) -> tuple[str, list[Image.Image], list[Image.Image], list[Image.Image], Image.Image]:
    """
    extract 3D_model (glb file) from a reconstructed scene

    To fit the project requirement, we extract the camera position and orientation from the scene as return value.

    parameters:
        silent: bool, if True, no print
        scene_state: SparseGAState, the scene state
        min_conf_thr: float, the minimum confidence threshold
        as_pointcloud: bool, if True, return the pointcloud
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
    outfile, renders, masks, combo_imgs = _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, delta=delta, max_inter_samples=max_inter_samples, silent=silent)
    return outfile, renders, masks, combo_imgs

def run_mast3r(
    outdir: str,
    gradio_delete_cache: bool,
    model: AsymmetricMASt3R,
    retrieval_model: str,
    device: torch.device,
    silent: bool,
    image_size: int,
    current_scene_state: str,
    filelist: list,
    optim_level: str,
    lr1: float,
    niter1: int,
    lr2: float,
    niter2: int,
    min_conf_thr: float,
    matching_conf_thr: float,
    as_pointcloud: bool,
    clean_depth: bool,
    transparent_cams: bool,
    cam_size: float,
    scenegraph_type: str,
    winsize: int,
    win_cyclic: bool,
    refid: int,
    TSDF_thresh: float,
    shared_intrinsics: bool,
    delta: float,
    max_inter_samples: int,
    **kw,
)->tuple[list[Image.Image], list[Image.Image], list[Image.Image], str, Image.Image, list[Image.Image]]:
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
        filelist = [filelist[0], filelist[0] + "_2"]

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    elif scenegraph_type == "retrieval":
        scene_graph_params.append(str(winsize))  # Na
        scene_graph_params.append(str(refid))  # k

    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append("noncyclic")
    scene_graph = "-".join(scene_graph_params)

    sim_matrix = None
    if "retrieval" in scenegraph_type:
        assert retrieval_model is not None
        retriever = Retriever(retrieval_model, backbone=model, device=device)
        with torch.no_grad():
            sim_matrix = retriever(filelist)

        # Cleanup
        del retriever
        torch.cuda.empty_cache()

    pairs = make_pairs(
        imgs,
        scene_graph=scene_graph,
        prefilter=None,
        symmetrize=True,
        sim_mat=sim_matrix,
    )
    if optim_level == "coarse":
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if (
        current_scene_state is not None
        and not current_scene_state.should_delete
        and current_scene_state.cache_dir is not None
    ):
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix="_cache", dir=outdir)
    else:
        cache_dir = os.path.join(outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(
        filelist,
        pairs,
        cache_dir,
        model,
        lr1=lr1,
        niter1=niter1,
        lr2=lr2,
        niter2=niter2,
        device=device,
        opt_depth="depth" in optim_level,
        shared_intrinsics=shared_intrinsics,
        matching_conf_thr=matching_conf_thr,
        **kw,
    )
    if (
        current_scene_state is not None
        and not current_scene_state.should_delete
        and current_scene_state.outfile_name is not None
    ):
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix="_scene.glb", dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    outfile, renders, masks, combo_imgs = get_3D_model_from_scene(
        silent,
        scene_state,
        min_conf_thr,
        as_pointcloud,
        clean_depth,
        transparent_cams,
        cam_size,
        TSDF_thresh,
        delta,
        max_inter_samples,
    )
    # Ensure the scene is reloaded correctly for rendering
    scene_file = trimesh.load(outfile, force="scene")

    # render basic view of the scene
    preview = scene_file.save_image(resolution=(512, 512), visible=False)
    preview = Image.open(BytesIO(preview))

    return renders, masks, scene_state.sparse_ga.imgs, outfile, preview, combo_imgs


# ————————————————————————————————————————————————————————————————————
#  Build UI (single page, 3 logical parts)
# ————————————————————————————————————————————————————————————————————


# ──────────────────────────────────────────────────────────────────────────────
#  FULL MASt3R widget tab  (to be placed right above build_smooth_tab call)
# ──────────────────────────────────────────────────────────────────────────────
def build_mast3r_tab(model: AsymmetricMASt3R, device: torch.device, masks_state: gr.State, images_state: gr.State, original_images_state: gr.State):
    # create a real temp-folder *path* and keep a reference so it
    # isn’t deleted while the demo is running
    _tmp_dir_obj = tempfile.TemporaryDirectory(suffix="_mon3tr_gradio_demo")
    cache_path   = _tmp_dir_obj.name          # ← string path
    # (optionally) keep the object somewhere global so Python
    # doesn’t garbage-collect it:
    build_mast3r_tab._tmp_dir_obj = _tmp_dir_obj

    os.makedirs(cache_path, exist_ok=True)
    # internal helper that actually launches the sparse-GA pipeline
    def _run_mast3r(
        files,  # list[gr.File]
        coarse_lr,
        coarse_iter,
        fine_lr,
        fine_iter,
        optim_lvl,
        match_thr,
        shared_K,
        sg_type,
        sg_win,
        sg_cyclic,
        sg_refid,
        delta,
        ksamp,
        min_conf_thr,
        cam_size,
        TSDF_thresh,
        as_pointcloud,
        clean_depth,
        transparent_cams,
    )->tuple[list[Image.Image], list[Image.Image], str, Image.Image, list[Image.Image]]:
        if not files:
            raise gr.Error("Please drop at least one image.")
        paths = [f.name for f in files]

        # prepare model-side kwargs ------------------------------------------------
        lr1, niter1 = float(coarse_lr), int(coarse_iter)
        lr2, niter2 = float(fine_lr), int(fine_iter)
        optim_level = optim_lvl
        matching_conf_thr = float(match_thr)

        # sparse-GA call ----------------------------------------------------------
        renders, masks, imgs_raw, rec_model, preview, combo_imgs = run_mast3r(
            outdir=cache_path,
            gradio_delete_cache=True,
            model=model,
            retrieval_model=None,
            device=device,
            silent=True,
            image_size=512,
            current_scene_state=None,
            filelist=paths,
            optim_level=optim_level,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            matching_conf_thr=matching_conf_thr,
            shared_intrinsics=shared_K,
            scenegraph_type=sg_type,
            winsize=sg_win,
            win_cyclic=sg_cyclic,
            refid=sg_refid,
            delta=float(delta),
            max_inter_samples=int(ksamp),
            min_conf_thr=float(min_conf_thr),
            cam_size=float(cam_size),
            TSDF_thresh=float(TSDF_thresh),
            as_pointcloud=as_pointcloud,
            clean_depth=clean_depth,
            transparent_cams=transparent_cams,
        )
        # store masks so the Smooth tab can see them
        return renders, masks, imgs_raw, rec_model, preview, combo_imgs

    silent = True
    retrieval_model = None

    if not silent:
        print("Outputing stuff in", cache_path)

    available_scenegraph_type = [
        ("complete: all possible image pairs", "complete"),
        ("swin: sliding window", "swin"),
        ("logwin: sliding window with long range", "logwin"),
        ("oneref: match one image with all", "oneref"),
    ]
    if retrieval_model is not None:
        available_scenegraph_type.insert(
            1, ("retrieval: connect views based on similarity", "retrieval")
        )

    gr.Markdown("### 3D scene reconstruction with MASt3R")
    # ============  1.  INPUT ⇢ optimisation parameters  =====================
    with gr.Column():
        f_in = gr.File(file_count="multiple", label="Images (min 1)")
        run_bt = gr.Button("Run MASt3R", variant="primary")

    with gr.Row():
        with gr.Accordion("Optimisation hyper-params", open=False):
            with gr.Row():
                coarse_lr = gr.Slider(0.001, 0.2, 0.07, label="Coarse LR")
                coarse_it = gr.Slider(0, 1000, 300, label="Coarse iters", step=1)
            with gr.Row():
                fine_lr = gr.Slider(0.0005, 0.05, 0.01, label="Fine LR")
                fine_it = gr.Slider(0, 1000, 300, label="Fine iters", step=1)
            optim_lvl = gr.Dropdown(
                ["coarse", "refine", "refine+depth"],
                value="refine+depth",
                label="Opt level",
            )

        with gr.Accordion("Matching & scene-graph", open=False):
            match_thr = gr.Slider(0, 30, 0, label="Matching-conf thr")
            shared_K = gr.Checkbox(label="Shared intrinsics")
            sg_type = gr.Dropdown(
                ["complete", "swin", "logwin", "oneref"],
                value="complete",
                label="Scene-graph type",
            )
            sg_win = gr.Slider(1, 20, 1, step=1, label="win-size (swin/logwin)")
            sg_cyclic = gr.Checkbox(value=False, label="Cyclic")
            sg_refid = gr.Slider(0, 20, 0, step=1, label="ref-id (oneref)")

        # ============  2.  Interp. parameters  ==================================
        with gr.Accordion("Camera interpolation", open=False):
            delta = gr.Slider(0, 1, 0.5, label="Interpolation offsets")
            ksamp = gr.Slider(1, 10, 3, step=1, label="Interpolation max samples")

    # ============  3.  Post-processing parameters  ==================================
    with gr.Row():
        # adjust the confidence threshold
        min_conf_thr = gr.Slider(
            label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1
        )
        # adjust the camera size in the output pointcloud
        cam_size = gr.Slider(
            label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001
        )
        TSDF_thresh = gr.Slider(
            label="TSDF Threshold", value=0.0, minimum=0.0, maximum=1.0, step=0.01
        )
    with gr.Row():
        as_pointcloud = gr.Checkbox(value=False, label="As pointcloud")
        # two post process implemented
        clean_depth = gr.Checkbox(value=True, label="Clean-up depthmaps")
        transparent_cams = gr.Checkbox(value=True, label="Transparent cameras")
    # ============  3.  OUTPUTS  =============================================
    with gr.Row():
        out_model = gr.Model3D(label="Model", height="auto")
        out_image = gr.Image(label="Rendered view", height="auto")
    out_gallery = gr.Gallery(label="Intermediate renders and masks", columns=3, height="auto")

    # ============  4.  Wire everything ======================================
    run_bt.click(
        _run_mast3r,
        inputs=[
            f_in,
            coarse_lr,
            coarse_it,
            fine_lr,
            fine_it,
            optim_lvl,
            match_thr,
            shared_K,
            sg_type,
            sg_win,
            sg_cyclic,
            sg_refid,
            delta,
            ksamp,
            min_conf_thr,
            cam_size,
            TSDF_thresh,
            as_pointcloud,
            clean_depth,
            transparent_cams,
        ],
        outputs=[
            images_state,
            masks_state,  # goes to Smooth-Diffusion tab through State
            original_images_state,
            out_model,  # shown locally
            out_image,  # shown locally
            out_gallery,
        ],  # shown locally
    )


def build_smooth_tab(
    wrapper: SmoothWrapper, masks_state: gr.State, images_state: gr.State, inpaint_state: gr.State
):
    gr.Markdown("### Smooth‑Diffusion in‑painting")
    with gr.Row():
        g_imgs = gr.Gallery(label="Source", columns=4, type="pil")
        g_msks = gr.Gallery(label="Masks", columns=4, type="pil")

    with gr.Row():
        with gr.Column():
            prm = gr.Textbox(label="Prompt", value="ultra‑realistic, 8k render")

            with gr.Accordion("Model", open=True):
                dff = gr.Dropdown(
                    list(choices["diffuser"]),
                    value=default["diffuser"],
                    label="Diffuser",
                )
                lra = gr.Dropdown(
                    list(choices["lora"]), value=default["lora"], label="LoRA"
                )
                sch = gr.Dropdown(
                    list(choices["scheduler"]),
                    value=default["scheduler"],
                    label="Scheduler",
                )

        with gr.Column():
            cfg = gr.Slider(1, 15, value=default["cfg_scale"], label="CFG scale")
            strn = gr.Slider(
                0, 1, value=default["strength"], step=0.01, label="Strength"
            )
            step = gr.Slider(10, 100, value=default["step"], step=1, label="Steps")
            rad = gr.Slider(
                1, 100, value=default["blur_radius"], step=1, label="Blur radius"
            )

    run = gr.Button("Run in‑paint")

    g_blur = gr.Gallery(label="Blurred", columns=4)
    g_out = gr.Gallery(label="In‑painted", columns=4)

    # — model switchers —
    dff.change(
        lambda d, l, s: wrapper.load_all(d, l, s), inputs=[dff, lra, sch], outputs=[]
    )
    lra.change(
        lambda d, l, s: wrapper.load_all(d, l, s), inputs=[dff, lra, sch], outputs=[]
    )
    sch.change(lambda s: wrapper._load_scheduler(s), inputs=sch, outputs=[])

    # — propagate masks from reconstruction —
    def _on_masks_change(masks, imgs):
        return masks, imgs

    masks_state.change(
        _on_masks_change, inputs=[masks_state, images_state], outputs=[g_msks,g_imgs]
    )
    images_state.change(
        _on_masks_change, inputs=[masks_state, images_state], outputs=[g_msks,g_imgs]
    )

    # — run in‑paint —
    run.click(
        wrapper.run,
        inputs=[g_imgs, g_msks, prm, cfg, strn, step, rad],
        outputs=[g_blur, g_out, inpaint_state],
    )

def build_remast3r_tab(model: AsymmetricMASt3R, device: torch.device, inpaint_state: gr.State, images_state: gr.State):
    # create a real temp-folder *path* and keep a reference so it
    # isn’t deleted while the demo is running
    _tmp_dir_obj = tempfile.TemporaryDirectory(suffix="_mon3tr_gradio_demo")
    cache_path   = _tmp_dir_obj.name          # ← string path
    # (optionally) keep the object somewhere global so Python
    # doesn’t garbage-collect it:
    build_remast3r_tab._tmp_dir_obj = _tmp_dir_obj
    # merge the inpaint and images state
    image_gallery = merge_list_states(inpaint_state, images_state)

    os.makedirs(cache_path, exist_ok=True)
    # internal helper that actually launches the sparse-GA pipeline
    def _run_remast3r(
        image_gallery,  # list[Image.Image]
        coarse_lr,
        coarse_iter,
        fine_lr,
        fine_iter,
        optim_lvl,
        match_thr,
        shared_K,
        sg_type,
        sg_win,
        sg_cyclic,
        sg_refid,
        delta,
        ksamp,
        min_conf_thr,
        cam_size,
        TSDF_thresh,
        as_pointcloud,
        clean_depth,
        transparent_cams,
    )->tuple[list[Image.Image], list[Image.Image], str, Image.Image, list[Image.Image]]:

        # prepare model-side kwargs ------------------------------------------------
        lr1, niter1 = float(coarse_lr), int(coarse_iter)
        lr2, niter2 = float(fine_lr), int(fine_iter)
        optim_level = optim_lvl
        matching_conf_thr = float(match_thr)
        # save images to temp folder
        filelist = []
        for i, img in enumerate(image_gallery):
            img[0].save(os.path.join(cache_path, f"final_img_{i}.png"))
            filelist.append(os.path.join(cache_path, f"final_img_{i}.png"))
        # sparse-GA call ----------------------------------------------------------
        scene, outfile, imgs, basic_img = get_reconstructed_scene(
            outdir=cache_path,
            gradio_delete_cache=True,
            model=model,
            retrieval_model=None,
            device=device,
            silent=True,
            image_size=512,
            current_scene_state=None,
            filelist=filelist,
            optim_level=optim_level,
            lr1=lr1,
            niter1=niter1,
            lr2=lr2,
            niter2=niter2,
            matching_conf_thr=matching_conf_thr,
            shared_intrinsics=shared_K,
            scenegraph_type=sg_type,
            winsize=sg_win,
            win_cyclic=sg_cyclic,
            refid=sg_refid,
            delta=float(delta),
            max_inter_samples=int(ksamp),
            min_conf_thr=float(min_conf_thr),
            cam_size=float(cam_size),
            TSDF_thresh=float(TSDF_thresh),
            as_pointcloud=as_pointcloud,
            mask_sky=False, # does not matter but need to fill it.
            clean_depth=clean_depth,
            transparent_cams=transparent_cams,
        )
        # store masks so the Smooth tab can see them
        return outfile, basic_img, imgs

    silent = True
    retrieval_model = None

    if not silent:
        print("Outputing stuff in", cache_path)

    available_scenegraph_type = [
        ("complete: all possible image pairs", "complete"),
        ("swin: sliding window", "swin"),
        ("logwin: sliding window with long range", "logwin"),
        ("oneref: match one image with all", "oneref"),
    ]
    if retrieval_model is not None:
        available_scenegraph_type.insert(
            1, ("retrieval: connect views based on similarity", "retrieval")
        )

    gr.Markdown("### ReMASt3R: 3D scene reconstruction with inpainted images")
    # ============  1.  INPUT ⇢ optimisation parameters  =====================
    with gr.Column():
        image_gallery = gr.Gallery(label="Inpainted images with original images", columns=8, type="pil")
        run_bt = gr.Button("Run Remast3r", variant="primary")

    with gr.Row():
        with gr.Accordion("Optimisation hyper-params", open=False):
            with gr.Row():
                coarse_lr = gr.Slider(0.001, 0.2, 0.07, label="Coarse LR")
                coarse_it = gr.Slider(0, 1000, 300, label="Coarse iters", step=1)
            with gr.Row():
                fine_lr = gr.Slider(0.0005, 0.05, 0.01, label="Fine LR")
                fine_it = gr.Slider(0, 1000, 300, label="Fine iters", step=1)
            optim_lvl = gr.Dropdown(
                ["coarse", "refine", "refine+depth"],
                value="refine+depth",
                label="Opt level",
            )

        with gr.Accordion("Matching & scene-graph", open=False):
            match_thr = gr.Slider(0, 30, 0, label="Matching-conf thr")
            shared_K = gr.Checkbox(label="Shared intrinsics")
            sg_type = gr.Dropdown(
                ["complete", "swin", "logwin", "oneref"],
                value="complete",
                label="Scene-graph type",
            )
            sg_win = gr.Slider(1, 20, 1, step=1, label="win-size (swin/logwin)")
            sg_cyclic = gr.Checkbox(value=False, label="Cyclic")
            sg_refid = gr.Slider(0, 20, 0, step=1, label="ref-id (oneref)")

        # ============  2.  Interp. parameters  ==================================
        with gr.Accordion("Camera interpolation", open=False):
            delta = gr.Slider(0, 1, 0.5, label="Interpolation offsets")
            ksamp = gr.Slider(1, 10, 3, step=1, label="Interpolation max samples")

    # ============  3.  Post-processing parameters  ==================================
    with gr.Row():
        # adjust the confidence threshold
        min_conf_thr = gr.Slider(
            label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1
        )
        # adjust the camera size in the output pointcloud
        cam_size = gr.Slider(
            label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001
        )
        TSDF_thresh = gr.Slider(
            label="TSDF Threshold", value=0.0, minimum=0.0, maximum=1.0, step=0.01
        )
    with gr.Row():
        as_pointcloud = gr.Checkbox(value=False, label="As pointcloud")
        # two post process implemented
        clean_depth = gr.Checkbox(value=True, label="Clean-up depthmaps")
        transparent_cams = gr.Checkbox(value=True, label="Transparent cameras")
    # ============  3.  OUTPUTS  =============================================
    with gr.Row():
        out_model = gr.Model3D(label="Model", height="auto")
        out_image = gr.Image(label="Rendered view", height="auto")
    out_gallery = gr.Gallery(label="Final images compared with original images", columns=4, height="auto")

    def _on_images_change(inpaint_state, original_images_state):
        return inpaint_state + original_images_state
    # ============  4.  Wire everything ======================================
    inpaint_state.change(
        _on_images_change, inputs=[inpaint_state, images_state], outputs=[image_gallery]
    )
    images_state.change(
        _on_images_change, inputs=[inpaint_state, images_state], outputs=[image_gallery]
    )
    run_bt.click(
        _run_remast3r,
        inputs=[
            image_gallery,
            coarse_lr,
            coarse_it,
            fine_lr,
            fine_it,
            optim_lvl,
            match_thr,
            shared_K,
            sg_type,
            sg_win,
            sg_cyclic,
            sg_refid,
            delta,
            ksamp,
            min_conf_thr,
            cam_size,
            TSDF_thresh,
            as_pointcloud,
            clean_depth,
            transparent_cams,
        ],
        outputs=[
            out_model,  # shown locally
            out_image,  # shown locally
            out_gallery,
        ],  # shown locally
    )

# ————————————————————————————————————————————————————————————————————
#  Launch everything
# ————————————————————————————————————————————————————————————————————


def launch_demo(share=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    ).to(device)

    wrapper = SmoothWrapper(fp16=True)

    with gr.Blocks(title=VERSION, css=".gradio-container {min-width:100%}") as demo:
        gr.Markdown(f"# {VERSION}")

        original_images_state = gr.State([])  # holds list of PIL images
        interpolation_masks_state = gr.State([])  # holds list of PIL masks
        interpolation_images_state = gr.State([])  # holds list of PIL images
        inpaint_state = gr.State([])  # holds list of PIL images

        build_mast3r_tab(model, device, interpolation_masks_state, interpolation_images_state, original_images_state)
        build_smooth_tab(wrapper, interpolation_masks_state, interpolation_images_state,inpaint_state)
        build_remast3r_tab(model, device, inpaint_state, original_images_state)
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=share)


# ————————————————————————————————————————————————————————————————————
if __name__ == "__main__":
    launch_demo()
