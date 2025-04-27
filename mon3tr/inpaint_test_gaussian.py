################################################################################
# Smooth-Diffusion Demo v1.5  –  In-painting, FP16, blurred-mask preview
################################################################################
import gradio as gr, os, os.path as osp
from collections import OrderedDict
import torch, numpy as np
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    DDIMScheduler
)
from huggingface_hub import snapshot_download

# ─── Paths & flags ────────────────────────────────────────────────────────────
HERE              = osp.dirname(__file__)
LOCAL_MODELS_ROOT = osp.join(HERE, "../smooth-diffusion/assets/models")
LOW_VRAM, OFFLOAD = True, None
version           = "Smooth Diffusion Demo v1.5  –  blurred-mask preview"

# ─── Model choices ────────────────────────────────────────────────────────────
choices           = OrderedDict()
choices.diffuser  = OrderedDict([
    ("SD-v1-5", "runwayml/stable-diffusion-v1-5"),
    ("OJ-v4",   "prompthero/openjourney-v4"),
    ("RR-v2",   "SG161222/Realistic_Vision_V2.0"),
])
choices.lora      = OrderedDict([
    ("empty", ""),
    ("Smooth-LoRA-v1",
     osp.join(HERE, "../smooth-diffusion/assets/models/smooth_lora.safetensors")),
])
choices.scheduler = OrderedDict([("DDIM", DDIMScheduler)])

default = dict(diffuser="RR-v2", scheduler="DDIM", lora="Smooth-LoRA-v1",
               step=20, cfg_scale=5.0, blur_radius=10.0)

# ─── helpers ──────────────────────────────────────────────────────────────────
def _ensure_local(repo_id):
    path = repo_id if osp.isdir(repo_id) else osp.join(
        LOCAL_MODELS_ROOT, repo_id.replace("/", "--"))
    if not osp.isdir(path):
        snapshot_download(repo_id, local_dir=path, resume_download=True)
    return path

def regulate(img: Image.Image):
    w,h = img.size
    return img.resize((round(w/64)*64, round(h/64)*64), Image.BILINEAR)

def to_pil(obj, mode):
    if isinstance(obj, tuple): obj = obj[0]
    if isinstance(obj, Image.Image):
        img = obj
    elif isinstance(obj, np.ndarray):
        img = Image.fromarray(obj)
    elif isinstance(obj, str):
        img = Image.open(obj)
    else:
        raise TypeError(type(obj))
    return img.convert(mode)

def blur_mask(mask: Image.Image, radius: float):
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))

# ─── Wrapper ──────────────────────────────────────────────────────────────────
class Wrapper:
    def __init__(self, fp16=True, tag_diffuser=None, tag_lora=None, tag_scheduler=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype  = torch.float16 if fp16 else torch.float32
        self.net = self.inpaint_net = None
        self.tag_diffuser = self.tag_lora = self.tag_scheduler = None
        self.load_all(tag_diffuser, tag_lora, tag_scheduler)

    # ---------------- model loading -----------------------------------------
    def load_all(self, d, l, s):
        self._load_diffuser_lora(d, l)
        self._load_scheduler(s)
        return d, l, s

    def _load_diffuser_lora(self, d, l):
        if self.net and (d, l) == (self.tag_diffuser, self.tag_lora):
            return
        model_dir = _ensure_local(choices.diffuser[d])
        self.net = StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=torch.float16,
            low_cpu_mem_usage=True, use_safetensors=None
        ).to(self.device)

        if LOW_VRAM:
            self.net.enable_attention_slicing()
            self.net.enable_vae_tiling()

        self.net.safety_checker = None
        if l != "empty":
            l_path = choices.lora[l]
            if l_path.endswith((".bin", ".safetensors")):
                self.net.unet.load_attn_procs(osp.dirname(l_path),
                                              weight_name=osp.basename(l_path))
            else:
                self.net.unet.load_attn_procs(l_path)
            self.net.unet.to(self.device)

        self.tag_diffuser, self.tag_lora = d, l
        self.inpaint_net = None          # force rebuild

    def _load_scheduler(self, s):
        self.net.scheduler = choices.scheduler[s].from_config(self.net.scheduler.config)
        self.tag_scheduler = s
        if self.inpaint_net:
            self.inpaint_net.scheduler = self.net.scheduler
        return s

    # ---------------- in-paint pipeline -------------------------------------
    def _get_inpaint(self):
        if self.inpaint_net:
            return self.inpaint_net
        base_id = choices.diffuser[self.tag_diffuser]
        self.inpaint_net = StableDiffusionInpaintPipeline.from_pretrained(
            base_id, torch_dtype=self.net.unet.dtype,
            low_cpu_mem_usage=True, use_safetensors=None
        ).to(self.device)

        # share modules
        self.inpaint_net.unet = self.net.unet
        self.inpaint_net.vae  = self.net.vae
        self.inpaint_net.safety_checker = None
        self.inpaint_net.scheduler      = self.net.scheduler

        if self.tag_lora != "empty":
            l_path = choices.lora[self.tag_lora]
            if l_path.endswith((".bin", ".safetensors")):
                self.inpaint_net.unet.load_attn_procs(
                    osp.dirname(l_path), weight_name=osp.basename(l_path))
            else:
                self.inpaint_net.unet.load_attn_procs(l_path)
            self.inpaint_net.unet.to(self.device)
        return self.inpaint_net

    # ---------------- main call --------------------------------------------
    @torch.no_grad()
    def run(self, imgs, masks, prompt, cfg, step, radius, d, l, s):
        if (d, l) != (self.tag_diffuser, self.tag_lora):
            self.load_all(d, l, s)
        elif s != self.tag_scheduler:
            self._load_scheduler(s)
        pipe = self._get_inpaint()

        imgs  = [to_pil(i, "RGB") for i in imgs]
        masks = [to_pil(m, "L")   for m in masks]
        if len(imgs) != len(masks):
            raise ValueError("images and masks counts differ")

        blurred_masks, outputs = [], []
        for img, msk in zip(imgs, masks):
            img = regulate(img)
            msk = regulate(msk)
            msk = blur_mask(msk, radius)
            blurred_masks.append(msk)

            out = pipe(prompt, img, msk,
                       guidance_scale=cfg,
                       num_inference_steps=step).images[0]
            outputs.append(out)
        return blurred_masks, outputs

# ─── UI ──────────────────────────────────────────────────────────────────────
def build_ui(w: Wrapper):
    with gr.Row():
        g_imgs = gr.Gallery(label="Images", type="pil", columns=4)
        g_msks = gr.Gallery(label="Masks",  type="pil", columns=4)
    g_blur = gr.Gallery(label="Blurred masks (preview)", columns=4)
    g_out  = gr.Gallery(label="Inpainted", columns=4)

    prm = gr.Textbox(label="Prompt")
    cfg = gr.Slider(1, 15, value=default["cfg_scale"], label="CFG scale")
    stp = gr.Slider(10, 100, value=default["step"], label="Steps")
    rad = gr.Slider(1, 100, value=default["blur_radius"], step=1,
                    label="Gaussian blur radius (px)")

    dff = gr.Dropdown(list(choices.diffuser), value=default["diffuser"], label="Diffuser")
    lra = gr.Dropdown(list(choices.lora),    value=default["lora"],    label="LoRA")
    sch = gr.Dropdown(list(choices.scheduler), value=default["scheduler"], label="Scheduler")

    run = gr.Button("Run")

    dff.change(w.load_all, [dff, lra, sch], [dff, lra, sch])
    lra.change(w.load_all, [dff, lra, sch], [dff, lra, sch])
    sch.change(w._load_scheduler, [sch], [sch])

    run.click(
        fn=w.run,
        inputs=[g_imgs, g_msks, prm, cfg, stp, rad, dff, lra, sch],
        outputs=[g_blur, g_out]
    )

# ─── launch ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wrapper = Wrapper(fp16=True,
                      tag_diffuser=default["diffuser"],
                      tag_lora=default["lora"],
                      tag_scheduler=default["scheduler"])

    with gr.Blocks(title=version) as demo:
        gr.Markdown(f"# {version}")
        with gr.Tab("In-painting"):
            build_ui(wrapper)

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
