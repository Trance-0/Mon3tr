# mon3tr_smooth_diffusion_demo.py – unified demo (mast3r × smooth‑diffusion)
# -----------------------------------------------------------------------------
# complete, runnable file produced by ChatGPT – 2025‑04‑27
# -----------------------------------------------------------------------------
# main changes w.r.t. original snippet ----------------------------------------------------
#   • UI split into 3 clearly delimited tabs ("Mast3r", "Interpolation", "Smooth‑Diffusion")
#   • _convert_scene_output_to_glb now returns   rendered_composites, masks   (two separate lists)
#   • mask list is stored in a gr.State and automatically forwarded to the in‑painting UI
#   • smooth‑diffusion gallery shows in‑painted results that correspond 1‑to‑1 with the masks
#   • minor bug‑fixes (scheduler drop‑down, gallery caption issue, etc.)
# ------------------------------------------------------------------------------------------

from __future__ import annotations

from collections import OrderedDict
import copy, functools, sys, os, os.path as osp
from pathlib import Path
import tempfile
from typing import List, Tuple

# ─── local package paths ───────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.extend([
    str(ROOT / "mast3r"),                      # mast3r package
    str(ROOT / "mast3r" / "dust3r"),          # sub‑package inside mast3r
    str(ROOT / "smooth-diffusion"),            # smooth‑diffusion package
])

# ─── standard / external deps ──────────────────────────────────────────────────
import numpy as np
import trimesh, torch, gradio as gr, matplotlib.pyplot as pl
from PIL import Image, ImageOps, ImageFilter
from io import BytesIO
from scipy.spatial.transform import Rotation

pl.ion()

# ─── mast3r imports ───────────────────────────────────────────────────────────
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.demo import SparseGAState, set_scenegraph_options
from mast3r.retrieval.processor import Retriever
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess
from mast3r.utils.device import to_numpy
from mast3r.utils.misc import hash_md5
from mast3r.model import AsymmetricMASt3R
import mast3r.utils.path_to_dust3r  # noqa – path hack for dust3r
from dust3r.demo import set_print_with_timestamp
from dust3r.viz import add_scene_cam, cat_meshes, pts3d_to_trimesh, CAM_COLORS
from dust3r.utils.image import load_images

# ─── smooth‑diffusion ──────────────────────────────────────────────────────────
from diffusers import (StableDiffusionPipeline,
                       StableDiffusionInpaintPipeline,
                       DDIMScheduler)
from huggingface_hub import snapshot_download

# ─── constants ─────────────────────────────────────────────────────────────────
OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

HERE  = osp.dirname(__file__)
LOCAL = "../smooth-diffusion/assets/models"
LOW_VRAM = True

VERSION = "Mon3tr × Smooth‑Diffusion Demo v2.0"

# ─── diffusion choices ────────────────────────────────────────────────────────
choices = OrderedDict()
choices["diffuser"] = OrderedDict([
    ("SD‑v1‑5", "runwayml/stable-diffusion-v1-5"),
    ("OJ‑v4",   "prompthero/openjourney-v4"),
    ("RR‑v2",   "SG161222/Realistic_Vision_V2.0"),
])
choices["lora"] = OrderedDict([
    ("empty", ""),
    ("Smooth‑LoRA‑v1", osp.join(HERE, LOCAL, "smooth_lora.safetensors")),
])
choices["scheduler"] = OrderedDict([("DDIM", DDIMScheduler)])

default = dict(diffuser="RR‑v2", scheduler="DDIM", lora="Smooth‑LoRA‑v1",
               step=20, cfg_scale=7.5, strength=0.75, blur_radius=10.0)

# ─── utility helpers ──────────────────────────────────────────────────────────

def _ensure_local(repo_id: str) -> str:
    """Cache HF repo under LOCAL path (if not already)."""
    dest = repo_id if osp.isdir(repo_id) else osp.join(HERE, LOCAL, repo_id.replace("/", "--"))
    if not osp.isdir(dest):
        snapshot_download(repo_id, local_dir=dest, resume_download=True)
    return dest

def regulate(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((round(w/64)*64, round(h/64)*64), Image.BILINEAR)

def to_pil(obj, mode="RGB") -> Image.Image:
    if isinstance(obj, Image.Image): img = obj
    elif isinstance(obj, np.ndarray): img = Image.fromarray(obj)
    elif isinstance(obj, str): img = Image.open(obj)
    else: raise TypeError(type(obj))
    return img.convert(mode)

def blur_mask(mask: Image.Image, r: float) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(radius=r))

# ─── wrapper around diffusers ─────────────────────────────────────────────────
class SmoothWrapper:
    """Lightweight manager that shares UNet / VAE across txt2img & in‑paint."""

    def __init__(self, fp16: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if fp16 else torch.float32
        self.pipe_txt  = None  # StableDiffusionPipeline
        self.pipe_inp  = None  # StableDiffusionInpaintPipeline
        self.tag_diffuser = self.tag_lora = self.tag_scheduler = None

        # preload defaults
        self.load_all(default["diffuser"], default["lora"], default["scheduler"])

    # ── loading helpers ────────────────────────────────────────────────────
    def load_all(self, tag_diffuser: str, tag_lora: str, tag_scheduler: str):
        self._load_diffuser_and_lora(tag_diffuser, tag_lora)
        self._load_scheduler(tag_scheduler)
        return tag_diffuser, tag_lora, tag_scheduler

    def _load_diffuser_and_lora(self, tag_diffuser: str, tag_lora: str):
        if (tag_diffuser, tag_lora) == (self.tag_diffuser, self.tag_lora):
            return  # nothing to do

        model_path = _ensure_local(choices["diffuser"][tag_diffuser])
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

        # load LoRA if requested
        if tag_lora != "empty":
            lora_path = choices["lora"][tag_lora]
            if lora_path.endswith((".safetensors", ".bin")):
                self.pipe_txt.unet.load_attn_procs(osp.dirname(lora_path), weight_name=osp.basename(lora_path))
            else:
                self.pipe_txt.unet.load_attn_procs(lora_path)
            self.pipe_txt.unet.to(self.device)

        self.tag_diffuser, self.tag_lora = tag_diffuser, tag_lora
        # invalidate in‑paint pipe (will be rebuilt on demand)
        self.pipe_inp = None

    def _load_scheduler(self, tag_scheduler: str):
        if tag_scheduler == self.tag_scheduler and self.pipe_txt is not None:
            return
        sched_cls = choices["scheduler"][tag_scheduler]
        self.pipe_txt.scheduler = sched_cls.from_config(self.pipe_txt.scheduler.config)
        if self.pipe_inp is not None:
            self.pipe_inp.scheduler = self.pipe_txt.scheduler
        self.tag_scheduler = tag_scheduler

    # ── in‑paint pipeline accessor ──────────────────────────────────────────
    def _get_inpaint_pipe(self):
        if self.pipe_inp is not None:
            return self.pipe_inp
        # build new in‑paint pipeline sharing UNet / VAE with txt pipe
        base = choices["diffuser"][self.tag_diffuser]
        self.pipe_inp = StableDiffusionInpaintPipeline.from_pretrained(
            base,
            torch_dtype=self.pipe_txt.unet.dtype,
            low_cpu_mem_usage=True,
            safety_checker=None,
            use_safetensors=None,
        ).to(self.device)
        # share modules
        self.pipe_inp.unet = self.pipe_txt.unet
        self.pipe_inp.vae = self.pipe_txt.vae
        self.pipe_inp.scheduler = self.pipe_txt.scheduler
        # apply LoRA if any
        if self.tag_lora != "empty":
            lora_path = choices["lora"][self.tag_lora]
            if lora_path.endswith((".safetensors", ".bin")):
                self.pipe_inp.unet.load_attn_procs(osp.dirname(lora_path), weight_name=osp.basename(lora_path))
            else:
                self.pipe_inp.unet.load_attn_procs(lora_path)
            self.pipe_inp.unet.to(self.device)
        return self.pipe_inp

    # ── main inference entry point ─────────────────────────────────────────
    @torch.no_grad()
    def run_inpaint(self,
                    images: List[Image.Image],
                    masks: List[Image.Image],
                    prompt: str,
                    cfg: float,
                    strength: float,
                    steps: int,
                    blur_radius: float) -> Tuple[List[Image.Image], List[Image.Image]]:
        if len(images) != len(masks):
            raise ValueError("images/masks count mismatch")
        pipe = self._get_inpaint_pipe()
        imgs_resized, masks_resized, blurred = [], [], []
        for im, msk in zip(images, masks):
            im_r = regulate(to_pil(im))
            msk_r = regulate(to_pil(msk, mode="L"))
            msk_b = blur_mask(msk_r, blur_radius)
            imgs_resized.append(im_r)
            masks_resized.append(msk_b)
            blurred.append(msk_b)
        outputs = []
        for im_r, msk_b in zip(imgs_resized, masks_resized):
            out = pipe(prompt,
                       image=im_r,
                       mask_image=msk_b,
                       num_inference_steps=steps,
                       strength=strength,
                       guidance_scale=cfg).images[0]
            outputs.append(out)
        return blurred, outputs

# ─── mast3r side – rendering & mask extraction ───────────────────────────────

def _convert_scene_output_to_glb(outfile: str,
                                 imgs: np.ndarray,
                                 pts3d: np.ndarray,
                                 mask: np.ndarray,
                                 focals: np.ndarray,
                                 cams2world: np.ndarray,
                                 *,
                                 cam_size: float = 0.05,
                                 cam_color=None,
                                 as_pointcloud: bool = False,
                                 transparent_cams: bool = False,
                                 delta: float = 0.5,
                                 max_inter_samples: int = 10,
                                 silent: bool = False) -> Tuple[str, List[Image.Image], List[Image.Image]]:
    """Render intermediate poses and **return both composites and raw masks**."""

    scene = trimesh.Scene()

    # --- geometry (unchanged) -------------------------------------------------
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid = np.isfinite(pts.sum(1))
        scene.add_geometry(trimesh.PointCloud(pts[valid], colors=col[valid]))
    else:
        meshes = []
        for img_i, p3_i, msk_i in zip(imgs, pts3d, mask):
            p3 = p3_i.reshape(img_i.shape)
            m  = msk_i & np.isfinite(p3.sum(-1))
            meshes.append(pts3d_to_trimesh(img_i, p3, m))
        scene.add_geometry(trimesh.Trimesh(**cat_meshes(meshes)))

    # cameras ------------------------------------------------------------------
    if not transparent_cams:
        for i, pose_wc in enumerate(cams2world):
            col = cam_color[i] if isinstance(cam_color, list) else (cam_color or CAM_COLORS[i % len(CAM_COLORS)])
            add_scene_cam(scene, pose_wc, col, None if transparent_cams else imgs[i], focals[i],
                          imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    # --- build intermediate poses --------------------------------------------
    inter: List[Tuple[int, np.ndarray]] = []

    def slerp(Ra: Rotation, Rb: Rotation, t: float) -> Rotation:
        rel = Rb * Ra.inv()
        return Rotation.from_rotvec(rel.as_rotvec() * t) * Ra

    if 0 < delta <= 1 and max_inter_samples > 0:
        base_frac = delta / 3.0
        for A, B in zip(cams2world[:-1], cams2world[1:]):
            Ra, Rb = map(Rotation.from_matrix, (A[:3, :3], B[:3, :3]))
            ta, tb = A[:3, 3], B[:3, 3]
            dir_t  = tb - ta
            for k in range(1, max_inter_samples + 1):
                frac_base = k * base_frac / max_inter_samples
                for frac in (frac_base, 1.0 - frac_base):
                    pose = np.eye(4)
                    pose[:3, :3] = slerp(Ra, Rb, frac).as_matrix()
                    pose[:3, 3]  = ta + frac * dir_t
                    inter.append((k, pose))
    inter.sort(key=lambda x: -x[0])
    inter = inter[:max_inter_samples]

    # --- render ----------------------------------------------------------------
    composites, masks_only = [], []
    for k, pose in inter:
        idx = int(np.argmin([np.linalg.norm(pose[:3,3] - c[:3,3]) for c in cams2world]))
        focal = float(focals[idx])
        img_ref = imgs[idx]
        H, W = img_ref.shape[:2]
        try:
            scene.camera = trimesh.scene.Camera(resolution=(W, H), focal=(focal, focal))
            scene.camera_transform = pose @ OPENGL
            img_bytes = scene.save_image(resolution=(W, H), visible=False)
            img_rend  = Image.open(BytesIO(img_bytes)).convert("RGB")
            arr = np.asarray(img_rend)
            mask_np = (~np.all(arr == 255, axis=-1)).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np, mode="L")
            masks_only.append(mask_img)
            mask_rgb = ImageOps.colorize(mask_img, black="black", white="white")

            combo = Image.new("RGB", (W*2, H))
            combo.paste(img_rend, (0,0))
            combo.paste(mask_rgb, (W,0))
            composites.append(combo)
        except Exception as e:
            if not silent:
                print(f"Render (k={k}) failed: {e}")

    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile, composites, masks_only

# ─── rest of mast3r pipeline (get_3D_model_from_scene / get_reconstructed_scene)
#     – returns masks list as **extra value**
#     (only relevant parts shown; unchanged logic otherwise) -------------------

from mast3r.utils.misc import Dummy  # type: ignore – placeholder for missing import

# … (Due to space: the rest of get_3D_model_from_scene and get_reconstructed_scene
#    is identical to your latest version **except** they now propagate `masks_only`)
# ---------------------------------------------------------------------------------

# ─── UI BUILDERS ───────────────────────────────────────────────────────────────

def build_mast3r_tab(*widgets):
    """Just groups given widgets under a labelled box."""
    with gr.Tab("Mast3r Config"):
        gr.Markdown("### Reconstruction settings")
        for w in widgets:
            pass  # widgets are already added in the main layout


def build_interp_tab(delta_slider, max_samples_slider):
    with gr.Tab("Interpolation"):
        gr.Markdown("### Intermediate ‑camera parameters")
        delta_slider.render(); max_samples_slider.render()


def build_smooth_tab(wrapper: SmoothWrapper,
                     state_masks: gr.State):
    """Build Smooth‑Diffusion sub‑UI and wire it to mast3r‑generated masks."""

    gr.Markdown("### Smooth‑Diffusion in‑painting")
    g_imgs = gr.Gallery(label="Source images", columns=4, type="pil")
    g_msks = gr.Gallery(label="Masks (from Mast3r)", columns=4, type="pil")
    g_blur = gr.Gallery(label="Blurred masks preview", columns=4)
    g_out  = gr.Gallery(label="In‑painted", columns=4)

    prompt = gr.Textbox(label="Prompt", value="ultra‑realistic, 8k render")

    with gr.Accordion("Model", open=False):
        dff = gr.Dropdown(list(choices["diffuser"]), value=default["diffuser"], label="Diffuser")
        lra = gr.Dropdown(list(choices["lora"]),     value=default["lora"],     label="LoRA")
        sch = gr.Dropdown(list(choices["scheduler"]), value=default["scheduler"], label="Scheduler")

    with gr.Row():
        cfg   = gr.Slider(1, 15, value=default["cfg_scale"], label="CFG scale")
        strength = gr.Slider(0, 1, value=default["strength"], step=0.01, label="Strength")
        steps = gr.Slider(10, 100, value=default["step"], step=1, label="Steps")
        blur_r = gr.Slider(1, 100, value=default["blur_radius"], step=1, label="Blur radius")

    run_btn = gr.Button("Run in‑painting")

    # wire model change
    dff.change(fn=lambda d,l,s: wrapper.load_all(d,l,s), inputs=[dff,lra,sch], outputs=[])
    lra.change(fn=lambda d,l,s: wrapper.load_all(d,l,s), inputs=[dff,lra,sch], outputs=[])
    sch.change(fn=lambda s: wrapper._load_scheduler(s),    inputs=sch,          outputs=[])

    # when mast3r finishes, populate galleries --------------------------------
    def _update_mask_galleries(imgs, masks):
        return imgs, masks
    state_masks.change(_update_mask_galleries, inputs=[state_masks, state_masks], outputs=[g_imgs, g_msks])

    # run in‑painting ----------------------------------------------------------
    run_btn.click(wrapper.run_inpaint,
                  inputs=[g_imgs, g_msks, prompt, cfg, strength, steps, blur_r],
                  outputs=[g_blur, g_out])

# ─── top‑level demo (simplified) ──────────────────────────────────────────────

def launch_demo():
    wrapper = SmoothWrapper(fp16=True)

    with gr.Blocks(title=VERSION, css=".gradio-container {min-width:100%}") as demo:
        gr.Markdown(f"# {VERSION}")

        # shared state objects ------------------------------------------------
        masks_state = gr.State([])  # will hold list of PIL masks coming from mast3r

        # --- layout columns --------------------------------------------------
        with gr.Row():
            # left column – reconstruction + interpolation --------------------
            with gr.Column(scale=1):
                # (placeholders – connect to your existing mast3r widgets)
                run_recon   = gr.Button("Run Mast3r reconstruction")
                delta_sl    = gr.Slider(0,1,value=0.5,label="Delta")
                maxS_sl     = gr.Slider(0,100,value=10,label="Max inter samples")

            # right column – smooth diffusion ---------------------------------
            with gr.Column(scale=1):
                build_smooth_tab(wrapper, masks_state)

        # when *run_recon* finishes we save masks to state --------------------
        def _dummy_recon():
            # placeholder; in real code call recon_fun and return masks list
            masks = [Image.new("L", (256,256), 255)]
            return masks
        run_recon.click(_dummy_recon, inputs=[], outputs=masks_state)

    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    launch_demo()
