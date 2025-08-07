from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import os

# 加载本地图像
image = Image.open("/workspace/测试文件夹/city.png")
model_path = r"/home/public/ai_chat_data/models/oneformer_cityscapes_swin_large"

# 设置环境变量，禁止 Hugging Face 尝试在线下载
# os.environ["HF_DATASETS_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 设置代理
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 加载模型和处理器
processor = OneFormerProcessor.from_pretrained(model_path, local_files_only=True)
model = OneFormerForUniversalSegmentation.from_pretrained(model_path, local_files_only=True)

# 语义分割
semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
semantic_outputs = model(**semantic_inputs)
# 后处理
predicted_semantic_map = processor.post_process_semantic_segmentation(semantic_outputs, target_sizes=[image.size[::-1]])[0]

# 实例分割
instance_inputs = processor(images=image, task_inputs=["instance"], return_tensors="pt")
instance_outputs = model(**instance_inputs)
# 后处理
predicted_instance_map = processor.post_process_instance_segmentation(instance_outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]

# 全景分割
panoptic_inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
panoptic_outputs = model(**panoptic_inputs)
# 后处理
predicted_panoptic_map = processor.post_process_panoptic_segmentation(panoptic_outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]

# 保存结果（可选）
predicted_semantic_map.save("/workspace/测试文件夹/semantic_result.png")
predicted_instance_map.save("/workspace/测试文件夹/instance_result.png")
predicted_panoptic_map.save("/workspace/测试文件夹/panoptic_result.png")
