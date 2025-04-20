import os
import torch
from mast3r.mast3r.model import AsymmetricMASt3R
from PIL import Image
import numpy as np
import gradio as gr

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

iface = gr.Interface(
    fn=process_images,
    inputs=[
        gr.inputs.File(type="file", label="Upload Images", multiple=True),
        gr.inputs.Textbox(default='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric', label="Model Weights"),
        gr.inputs.Textbox(default='cuda', label="Device")
    ],
    outputs=None,
    title="3D Image Generator",
    description="Upload images to generate and visualize a 3D representation using the MASt3R model."
)

if __name__ == '__main__':
    iface.launch()
