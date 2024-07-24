# Enhanced Stable Diffusion XL Image Generator

This project provides an interactive user interface to generate images using the Stable Diffusion XL model with custom LoRA weights. The interface is built using Gradio, allowing users to input prompts and adjust various parameters to control the image generation process.

![AN ANIME BOY](https://github.com/EchoSingh/Sacred-Backbench-AI-Creations/blob/main/pic1.jpg)


## Features

- **Prompt Input**: Enter text prompts to generate images.
- **Inference Steps**: Adjust the number of inference steps for image generation.
- **Guidance Scale**: Control the strength of the guidance.
- **Image Dimensions**: Set the width and height of the generated images.
- **Real-time Updates**: Live mode for immediate feedback.

## Prerequisites

- **CUDA**: A CUDA-compatible GPU is required for running this project efficiently. Ensure that you have the necessary CUDA drivers and libraries installed.

## Installation

To run this project, you need to install the required libraries. You can do this using the following commands:

```bash
pip install diffusers transformers torch accelerate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install peft --upgrade
pip install gradio
```
## Usage
1. Clone the Repository:

```bash
git clone https://github.com/EchoSingh/Sacred-Backbench-AI-Creations.git
cd Sacred-Backbench-AI-Creations
```
2. Run the Script:

```bash
python app.py
```
3. Access the Gradio Interface:

After running the script, you will see a local URL in the terminal. Open this URL in your browser to access the interactive interface.

## Example
Here’s how the interface looks:
![A BEAUTIFUL GIRL](https://github.com/EchoSingh/Sacred-Backbench-AI-Creations/blob/main/Pic2.png)

### Parameters
1. Prompt: Enter your text prompt here.
2. Inference Steps: Number of steps to run the diffusion process (1-100).
3. Guidance Scale: Scale for classifier-free guidance (1.0-20.0).
4. Width: Width of the generated image (256-1024).
5. Height: Height of the generated image (256-1024).

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/EchoSingh/Sacred-Backbench-AI-Creations/blob/main/LICENSE) file for details.

## Acknowledgments
1. Diffusers Library
2. Gradio Library
3. Stable Diffusion

