# ComfyUI-AstralAnimator
 A custom node for ComfyUI that enables smooth, keyframe-based animations for image generation. Create dynamic sequences with control over motion, zoom, rotation, and easing effects. Ideal for AI-assisted animation and video content creation.

![Astral](https://github.com/user-attachments/assets/af692fef-6b7c-4d7c-9ea9-1e73b102f0e7)

## Features

- Seamless integration with ComfyUI
- Keyframe-based animation system
- Control over X and Y motion, zoom, and rotation
- Various easing functions for smooth transitions
- Batch processing for efficient animation generation
- Compatible with different sampling methods and schedulers

## Installation

1. Ensure you have ComfyUI installed and set up on your system.
2. Clone this repository into your ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-AstralAnimator.git
   ```
3. Restart ComfyUI or reload custom nodes.

## Usage

1. In the ComfyUI interface, locate the "AstralAnimator" node under the "animation" category.
2. Connect the required inputs:
   - `model`: Your chosen AI model
   - `vae`: VAE for encoding/decoding
   - `positive`/`negative`: Your conditioning inputs
   - `image`: The starting image for your animation
3. Set the animation parameters:
   - `frame_count`: Number of frames to generate
   - `x_motion`/`y_motion`: Movement along X and Y axes
   - `zoom`: Zoom factor (use values around 1.0 for subtle effects)
   - `rotation`: Rotation angle per frame
   - `denoise_min`/`denoise_max`: Range for denoising strength
   - `easing_type`: Choose from various easing functions for smooth transitions
4. Connect the output to a display or save node to view or export your animation.

## Example Workflow

[Include a screenshot or diagram of an example ComfyUI workflow using AstralAnimator]

## Parameters Explanation

- **x_motion/y_motion**: Pixels to move per frame. Positive values move right/down, negative left/up.
- **zoom**: Values > 1 zoom in, < 1 zoom out. Use values close to 1 for subtle effects.
- **rotation**: Degrees to rotate per frame. Positive is clockwise, negative counterclockwise.
- **denoise_min/denoise_max**: Controls the strength of the AI's influence on each frame.
- **easing_type**: Determines how the motion accelerates or decelerates over time.

## Tips for Best Results

- Start with small values for motion, zoom, and rotation to achieve smooth animations.
- Experiment with different easing types to find the right feel for your animation.
- Use a higher frame count for smoother transitions, but be aware of increased processing time.
- The seed parameter can be used to maintain consistency across frames or create variations.

## Contributing

Contributions to ComfyUI-AstralAnimator are welcome! Please feel free to submit pull requests, create issues or spread the word.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Thanks to the ComfyUI team for creating an excellent platform for AI image generation.
- Inspired by the Deforum project for Stable Diffusion.

