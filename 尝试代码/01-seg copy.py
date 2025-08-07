from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import os
import numpy as np
import pandas as pd
import time
import cv2
from shapely.geometry import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon as ShapelyPolygon


class GenericMask:
    """
    Attribute:
        polygons (list[ndarray]): list[ndarray]: polygons for this mask.
            Each ndarray has format [x, y, x, y, ...]
        mask (ndarray): a binary mask
    """

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width

        if isinstance(mask_or_polygons, np.ndarray):  # 二值掩码
            assert mask_or_polygons.shape == (height, width), f"mask shape: {mask_or_polygons.shape}, target dims: {height}, {width}"
            self._mask = mask_or_polygons.astype("uint8")
            return

        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(mask_or_polygons, type(mask_or_polygons)))

    @property
    def mask(self):
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._has_holes

    def mask_to_polygons(self, mask):
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        mask = np.ascontiguousarray(mask)
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x + 0.5 for x in res if len(x) >= 6]
        return res, has_holes


# 代理端口7890
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model_path = r"/home/public/ai_chat_data/models/oneformer_ade20k_swin_large"
# load OneFormer fine-tuned on ADE20k for universal segmentation
processor = OneFormerProcessor.from_pretrained(model_path, local_files_only=True)
model = OneFormerForUniversalSegmentation.from_pretrained(model_path, local_files_only=True)
# 将模型移动到GPU
model = model.to(device)

image_path = "/workspace/测试文件夹/bedroom.jpg"
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
)[0]

# 将 predicted_semantic_map 转换为 PIL 图像

# ADE20k 数据集的颜色映射
def create_ade20k_label_colormap():
    """Creates a label colormap used in ADE20K segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    # 初始化颜色映射数组，ADE20k有151个类别（包括背景类0）
    colormap = np.zeros((150, 3), dtype=np.uint8)
    
    # 背景类为黑色 (0,0,0)
    # colormap[0] = np.array([0, 0, 0])
    
    # 读取CSV文件
    df = pd.read_csv("测试文件夹/ade20k.csv")
    
    # 解析颜色代码并填充colormap
    for idx, row in df.iterrows():
        color_str = row['Color_Code (R,G,B)']
        # 提取RGB值
        rgb = eval(color_str)  # 将字符串"(R,G,B)"转换为元组
        colormap[idx] = np.array(rgb)  # idx+1 因为背景类已经占用了索引0
    
    return colormap

t1 = time.time()
# 获取颜色映射
color_map = create_ade20k_label_colormap()
print(color_map.shape)  # 应该输出 (151, 3)

# 将张量转换为numpy数组
semantic_map_array = predicted_semantic_map.cpu().numpy()

# 创建彩色分割图
height, width = semantic_map_array.shape
colored_semantic_map = np.zeros((height, width, 3), dtype=np.uint8)

# 获取唯一的标签和它们的面积计数
labels, areas = np.unique(semantic_map_array, return_counts=True)
# 按面积从大到小排序
sorted_idxs = np.argsort(-areas).tolist()
labels = labels[sorted_idxs]

# 为每个类别填充对应的颜色
for label in filter(lambda l: l < len(color_map), labels):
    binary_mask = (semantic_map_array == label).astype(np.uint8)
    mask = GenericMask(binary_mask, height, width)
    
    alpha = 0.8  # 填充透明度
    edge_alpha = 0.9  # 边缘透明度，稍高于填充透明度以突出边缘
    
    # 获取当前标签的颜色
    fill_color = color_map[label].tolist()
    
    # 边缘颜色计算：稍微降低亮度以突出边缘
    edge_color = (np.array(fill_color) * 0.7).astype(np.uint8).tolist()
    
    line_width = max(int(min(height, width) / 500), 1)  # 根据图像尺寸动态调整线宽
    
    # 创建临时图像分别用于填充和边缘
    temp_fill = np.zeros((height, width, 3), dtype=np.uint8)
    temp_edge = np.zeros((height, width, 3), dtype=np.uint8)
    
    if not mask.has_holes:
        for segment in mask.polygons:
            segment = segment.reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(temp_fill, [segment], fill_color)
            cv2.polylines(temp_edge, [segment], True, edge_color, line_width, cv2.LINE_AA)
    else:
        temp_fill[binary_mask == 1] = fill_color
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(temp_edge, contours, -1, edge_color, line_width, cv2.LINE_AA)
    
    # 先混合填充颜色
    colored_semantic_map = cv2.addWeighted(
        colored_semantic_map, 1, temp_fill, alpha, 0
    )
    
    # 再混合边缘线条
    colored_semantic_map = cv2.addWeighted(
        colored_semantic_map, 1, temp_edge, edge_alpha, 0
    )
    
    print(f"处理标签 {label}，面积: {areas[sorted_idxs[list(labels).index(label)]]}")

t2 = time.time()
print(f"获取颜色映射时间: {t2 - t1} 秒")

# 转换为PIL图像
semantic_map_image = Image.fromarray(colored_semantic_map)
print(f"总时间: {time.time() - t0} 秒")

# 保存结果
output_path = "semantic_segmentation_result.png"
semantic_map_image.save(output_path)
print(f"分割结果已保存到: {output_path}")
print(f"分割结果大小: {semantic_map_image.size}")

# 创建并排显示的图像
combined_image = Image.new('RGB', (image.width * 2, image.height))
combined_image.paste(image, (0, 0))
combined_image.paste(semantic_map_image, (image.width, 0))
combined_image.save("combined_result.png")
print("原图和分割结果的对比图已保存到: combined_result.png")



