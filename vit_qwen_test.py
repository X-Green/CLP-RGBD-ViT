from torchvision.transforms import transforms
from PIL import Image
import h5py
import numpy as np
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
import io


# Step 1: Load the NYUV2 dataset (assuming you have nyu_depth_v2_labeled.mat)
file_path = "../../data/nyu_depth_v2/nyu_depth_v2_labeled.mat"
with h5py.File(file_path, 'r') as f:
    images = np.array(f['images'])  # RGB images: (1449,3,640,480)
    depths = np.array(f['depths'])  # Depth maps: (1449,640,480)

test_index = 233
rgb_array = images[test_index, :, :, :].transpose(1, 2, 0)  # (H, W, C) uint8
depth = depths[test_index]  # (H, W) float
# Convert RGB numpy array to PIL Image for processing
rgb_image = Image.fromarray(rgb_array).convert("RGB")

# Step 2: Load Qwen3-VL model and extract visual encoder
model_name = "Qwen/Qwen3-VL-8B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)
visual_encoder = model.model.visual
device = next(model.parameters()).device

# Step 3: Prepare the image for the visual encoder
# 替换 Step 3 的图像预处理部分
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": rgb_image,
                "min_pixels": 224*224,  # 调整这个值来控制分辨率
                "max_pixels": 224*224   # 保持相同值以固定尺寸
            }
        ]
    }
]

# 使用 process_vision_info 处理
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

# 准备输入
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(device)

# 获取处理后的 pixel_values 和 grid_thw
pixel_values = inputs['pixel_values'].to(torch.float16)
grid_thw = inputs['image_grid_thw']
grid_h = grid_thw[0, 1].item()
grid_w = grid_thw[0, 2].item()

print(f"Processed pixel_values shape: {pixel_values.shape}")
print(f"Grid THW: {grid_thw}")

#
# mean = processor.image_processor.image_mean  # e.g., [0.485, 0.456, 0.406]
# std = processor.image_processor.image_std   # e.g., [0.229, 0.224, 0.225]
#
# transform = transforms.Compose([
#     transforms.Resize((320, 480)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])
# pixel_values = transform(rgb_image).unsqueeze(0).to(device, dtype=torch.float16)  # (1, C, H, W)
# patch_size = visual_encoder.config.patch_size  # Usually 16
# _, _, height, width = pixel_values.shape
# h = height // patch_size
# w = width // patch_size
# grid_thw = torch.tensor([[1, h, w]], dtype=torch.long).to(device)  # For single static image

# Step 4: Run the visual encoder to get embeddings
with torch.no_grad():
    visual_outputs = visual_encoder(pixel_values, grid_thw=grid_thw)

# image_embeds = visual_outputs.image_embeds  # (1, num_patches+1, embed_dim)
# print(f"Image embeddings shape: {image_embeds.shape}")


import code
code.interact(local=locals())
