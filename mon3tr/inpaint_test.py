################################################################################
# Smooth-Diffusion Demo v1.1  –  In-painting, FP16, fixed device mismatch
################################################################################
import gradio as gr
import os, os.path as osp
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
)
from huggingface_hub import snapshot_download

# ─── Config ───────────────────────────────────────────────────────────────────
HERE               = osp.dirname(__file__)
LOCAL_MODELS_ROOT  = osp.join(HERE, "../smooth-diffusion/assets/models")

LOW_VRAM           = True            # attention slicing + VAE tiling
OFFLOAD            = None            # CPU-offload disabled (prevents device mix-ups)
version            = "Smooth Diffusion Demo v1.1  –  In-painting (FP16, fixed)"

choices           = OrderedDict()
choices.diffuser  = OrderedDict([
    ('SD-v1-5', "runwayml/stable-diffusion-v1-5"),
    ('OJ-v4'  , "prompthero/openjourney-v4"),
    ('RR-v2'  , "SG161222/Realistic_Vision_V2.0"),
])
choices.lora      = OrderedDict([
    ('empty', ''),
    ('Smooth-LoRA-v1',
     osp.join(HERE, "../smooth-diffusion/assets/models/smooth_lora.safetensors")),
])
choices.scheduler = OrderedDict([('DDIM', DDIMScheduler)])

default = dict(diffuser='RR-v2', scheduler='DDIM', lora='Smooth-LoRA-v1',
               step=20, cfg_scale=5.0)

# ─── helpers ──────────────────────────────────────────────────────────────────
def _repo_local(repo_id:str)->str:
    return repo_id if osp.isdir(repo_id) else osp.join(
        LOCAL_MODELS_ROOT, repo_id.replace('/', '--'))

def _ensure_local(repo_id:str)->str:
    path=_repo_local(repo_id)
    if not osp.isdir(path):
        snapshot_download(repo_id, local_dir=path, resume_download=True)
    return path

def regulate(pil):
    w,h=pil.size; w,h=(round(w/64)*64, round(h/64)*64)
    return pil.resize((w,h), Image.BILINEAR)

def to_pil(obj, mode):
    if isinstance(obj, tuple): obj=obj[0]
    if isinstance(obj, Image.Image): img=obj
    elif isinstance(obj, np.ndarray): img=Image.fromarray(obj)
    elif isinstance(obj, str): img=Image.open(obj)
    else: raise TypeError(type(obj))
    return img.convert(mode)

# ─── wrapper ──────────────────────────────────────────────────────────────────
class Wrapper:
    def __init__(self, fp16=True, tag_diffuser=None, tag_lora=None,
                 tag_scheduler=None):

        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.dtype =torch.float16 if fp16 else torch.float32

        self.net=self.inpaint_net=None
        self.tag_diffuser=self.tag_lora=self.tag_scheduler=None

        self.load_all(tag_diffuser, tag_lora, tag_scheduler)

    # ── loading helpers ───────────────────────────────────────────────────
    def load_all(self,diff_tag,lora_tag,sched_tag):
        self.load_diffuser_lora(diff_tag,lora_tag)
        self.load_scheduler(sched_tag)
        return diff_tag, lora_tag, sched_tag            # ← return 3 values

    def load_diffuser_lora(self,diff_tag,lora_tag):
        if self.net and (diff_tag,lora_tag)==(self.tag_diffuser,self.tag_lora):
            return
        model_dir=_ensure_local(choices.diffuser[diff_tag])
        self.net=StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            use_safetensors=None).to(self.device)

        if LOW_VRAM:
            self.net.enable_attention_slicing()
            self.net.enable_vae_tiling()

        self.net.safety_checker=None
        if lora_tag!='empty':
            lora=choices.lora[lora_tag]
            if lora.endswith(('.bin','.safetensors')):
                self.net.unet.load_attn_procs(osp.dirname(lora),
                                              weight_name=osp.basename(lora))
            else: self.net.unet.load_attn_procs(lora)
            self.net.unet.to(self.device)

        self.tag_diffuser,self.tag_lora=diff_tag,lora_tag
        self.inpaint_net=None

    def load_scheduler(self,sched_tag):
        self.net.scheduler=choices.scheduler[sched_tag].from_config(
            self.net.scheduler.config)
        self.tag_scheduler=sched_tag
        if self.inpaint_net:
            self.inpaint_net.scheduler=self.net.scheduler
        return sched_tag                               # ← return 1 value

    # ── get / build in-paint pipe ─────────────────────────────────────────
    def _get_inpaint(self):
        if self.inpaint_net: return self.inpaint_net
        base=choices.diffuser[self.tag_diffuser]
        self.inpaint_net=StableDiffusionInpaintPipeline.from_pretrained(
            base, torch_dtype=self.net.unet.dtype, low_cpu_mem_usage=True,
            use_safetensors=None).to(self.device)

        self.inpaint_net.unet=self.net.unet
        self.inpaint_net.vae =self.net.vae
        self.inpaint_net.safety_checker=None
        self.inpaint_net.scheduler=self.net.scheduler
        if self.tag_lora!='empty':
            lora=choices.lora[self.tag_lora]
            if lora.endswith(('.bin','.safetensors')):
                self.inpaint_net.unet.load_attn_procs(
                    osp.dirname(lora), weight_name=osp.basename(lora))
            else: self.inpaint_net.unet.load_attn_procs(lora)
            self.inpaint_net.unet.to(self.device)
        return self.inpaint_net

    # ── main call ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def run(self,imgs,masks,prompt,cfg,step,diff,lora,sched):
        if (diff,lora)!=(self.tag_diffuser,self.tag_lora):
            self.load_all(diff,lora,sched)
        elif sched!=self.tag_scheduler:
            self.load_scheduler(sched)
        pipe=self._get_inpaint()

        imgs =[to_pil(i,"RGB") for i in imgs]
        masks=[to_pil(m,"L")   for m in masks]
        if len(imgs)!=len(masks): raise ValueError("imgs ≠ masks len")

        outs=[]
        for im,msk in zip(imgs,masks):
            im,msk=regulate(im),regulate(msk)
            outs.append(pipe(prompt,im,msk,
                             guidance_scale=cfg,num_inference_steps=step
                             ).images[0])
        return outs

# ─── UI ──────────────────────────────────────────────────────────────────────
def ui(w):
    with gr.Row():
        g_img=gr.Gallery(label="Images",type='pil',columns=4)
        g_msk=gr.Gallery(label="Masks", type='pil',columns=4)
    g_out=gr.Gallery(label="Inpainted",columns=4)

    prm =gr.Textbox(label="Prompt")
    cfg =gr.Slider(1,15,value=default['cfg_scale'],label="CFG")
    stp =gr.Slider(10,100,value=default['step'],label="Steps")

    dff =gr.Dropdown(list(choices.diffuser),value=default['diffuser'],label="Diffuser")
    lra =gr.Dropdown(list(choices.lora   ),value=default['lora']    ,label="LoRA")
    sch =gr.Dropdown(list(choices.scheduler),value=default['scheduler'],label="Scheduler")

    run=gr.Button("Run")

    dff.change(w.load_all,[dff,lra,sch],[dff,lra,sch])
    lra.change(w.load_all,[dff,lra,sch],[dff,lra,sch])
    sch.change(w.load_scheduler,[sch],[sch])

    run.click(w.run,[g_img,g_msk,prm,cfg,stp,dff,lra,sch],g_out)

# ─── launch ──────────────────────────────────────────────────────────────────
if __name__=="__main__":
    w=Wrapper(fp16=True,
              tag_diffuser=default['diffuser'],
              tag_lora    =default['lora'],
              tag_scheduler=default['scheduler'])

    with gr.Blocks(title=version) as demo:
        gr.Markdown(f"# {version}")
        with gr.Tab("In-painting"): ui(w)

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
