import torch
from diffusers import DiffusionPipeline
import gradio as gr


pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(
    "ntc-ai/SDXL-LoRA-slider.sacred-geometry",
    weight_name="sacred geometry.safetensors",
    adapter_name="3d"
)


def generate_image(prompt, num_inference_steps, guidance_scale, width, height):
    generator = torch.Generator("cuda").manual_seed(42) 
    output = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator
    ).images[0]
    return output


interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.inputs.Textbox(lines=2, placeholder="Enter your prompt here...", label="Prompt"),
        gr.inputs.Slider(1, 100, step=1, value=50, label="Inference Steps"),
        gr.inputs.Slider(1.0, 20.0, step=0.1, value=7.5, label="Guidance Scale"),
        gr.inputs.Slider(256, 1024, step=64, value=512, label="Width"),
        gr.inputs.Slider(256, 1024, step=64, value=512, label="Height")
    ],
    outputs=gr.outputs.Image(type="pil", label="Generated Image"),
    title="Enhanced Stable Diffusion XL Image Generator",
    description="Generate images using Stable Diffusion XL with custom LoRA weights. Adjust the parameters for different results.",
    layout="vertical",
    live=True
)


interface.launch()
