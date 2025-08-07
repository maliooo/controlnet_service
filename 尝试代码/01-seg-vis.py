from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import torch  # pip install torch
import numpy as np
import time
# 添加可视化代码
from src.visualizer import Visualizer
import matplotlib.pyplot as plt  # pip install matplotlib
import numpy as np
from pathlib import Path


# 代理端口7890
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model_path = r"models/oneformer_ade20k_swin_large"
# load OneFormer fine-tuned on ADE20k for universal segmentation
processor = OneFormerProcessor.from_pretrained(model_path, local_files_only=True)
model = OneFormerForUniversalSegmentation.from_pretrained(model_path, local_files_only=True)
# 将模型移动到GPU
model = model.to(device)

image_path = "images/bedroom.jpg"
# 读取为RGB格式
image = Image.open(image_path)
print(f"原始图像大小: {image.size}")

# # 将PIL图像转换为numpy数组（RGB格式）
# image_np = np.array(image)
# # 转换为BGR格式
# image_bgr = image_np[:, :, ::-1]  # RGB转BGR
# # 转回PIL图像
# image = Image.fromarray(image_bgr)

t0 = time.time()

# Semantic Segmentation
inputs = processor(image, ["semantic"], return_tensors="pt")
# 将输入数据移动到GPU
inputs = {k: v.to(device) for k, v in inputs.items()}
print(inputs.keys())
print(inputs["pixel_values"].shape)

t1 = time.time()
with torch.no_grad():
    outputs = model(**inputs)  #
print(f"模型推理时间: {time.time() - t1} 秒")

# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# you can pass them to processor for semantic postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[(image.height, image.width)]
)[0].cpu()


# 创建一个简单的元数据类，用于存储类别信息
class SimpleMetadata:
    def __init__(self, num_classes):
        self.stuff_classes = [f"class_{i}" for i in range(num_classes)]
        # 为每个类别生成随机颜色
        self.stuff_colors = []
        for i in range(num_classes):
            # 生成鲜艳的随机颜色
            color = [np.random.randint(0, 255) for _ in range(3)]
            self.stuff_colors.append(color)

# 获取分割图中的类别数量
num_classes = predicted_semantic_map.max().item() + 1
metadata = SimpleMetadata(num_classes)

# 创建可视化器
visualizer = Visualizer(np.array(image), metadata=metadata)

# 绘制语义分割结果
vis_output = visualizer.draw_sem_seg(predicted_semantic_map, alpha=0.7, is_text=True)

# 获取可视化结果
result_img = vis_output.get_image()

# 保存结果
plt.figure(figsize=(12, 12))
plt.imshow(result_img)
plt.axis('off')

output_dir = Path(__file__).parent / "output"
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "segmentation_result.jpg", bbox_inches='tight', pad_inches=0)
plt.close()

print(f"总处理时间: {time.time() - t0} 秒")
print(f"分割结果已保存到: {output_dir / 'segmentation_result.jpg'}")

