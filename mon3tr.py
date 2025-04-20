import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'mast3r'))

import torch
from mast3r.model import AsymmetricMASt3R
from mast3r.demo import get_args_parser as mast3r_get_args_parser
from PIL import Image
import numpy as np
import gradio as gr
import tempfile
from contextlib import nullcontext


def get_args_parser():
    parser = mast3r_get_args_parser()
    # change defaults
    parser.prog = 'Mon3tr demo'
    return parser

def load_images(image_files):
    images = []
    for img_file in image_files:
        with Image.open(img_file) as img:
            images.append(np.array(img))
    return images

def generate_3d_image(model, images, device='cuda'):
    model.eval()
    with torch.no_grad():
        # Convert images to tensor and move to device
        image_tensors = [torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device) for img in images]
        # Generate 3D representation
        outputs = [model(img_tensor) for img_tensor in image_tensors]
        return outputs

def visualize_3d(outputs):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for output in outputs:
        # Assuming output contains 3D coordinates
        coords = output.cpu().numpy()
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    plt.show()

def process_images(image_files, model_weights, device='cuda'):
    # Load model
    model = AsymmetricMASt3R.from_pretrained(model_weights).to(device)
    
    # Load images
    images = load_images(image_files)
    
    # Generate 3D image
    outputs = generate_3d_image(model, images, device)
    
    # Visualize 3D image
    visualize_3d(outputs)

def get_context(tmp_dir):
    return tempfile.TemporaryDirectory(suffix='_mast3r_gradio_demo') if tmp_dir is None else nullcontext(tmp_dir)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    
    with get_context(args.tmp_dir) as tmpdirname:
        iface = gr.Interface(
            fn=process_images,
            inputs=[
                gr.File(label="Upload Images", file_count="multiple"),
                gr.Textbox(value=args.model_name, label="Model Weights"),
                gr.Textbox(value=args.device, label="Device")
            ],
            outputs=None,
            title="3D Image Generator",
            description="Upload images to generate and visualize a 3D representation using the MASt3R model."
        )
        iface.launch(share=args.share, server_name=args.server_name, server_port=args.server_port)
