from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import os
import numpy as np
import pandas as pd
import time
import cv2
from typing import Optional, List, Tuple
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
def resize_image(image, size):
    """将图片安装短边缩放到size，长边等比例缩放"""
    width, height = image.size
    if width < height:
        width, height = height, width
    return image.resize((int(width / height * size), size))

def post_process_semantic_segmentation(
        outputs, target_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]
        # segmentation = segmentation.cpu()

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                # resized_logits = torch.nn.functional.interpolate(
                #     segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                # )
                # semantic_map = resized_logits[0].argmax(dim=0)
                # semantic_segmentation.append(semantic_map)
                # 优化方案1：使用半精度和内存优化
                with torch.no_grad():
                    # with torch.cuda.amp.autocast():
                        seg_view = segmentation[idx].unsqueeze_(0)  # inplace操作
                        resized_logits = torch.nn.functional.interpolate(
                            # seg_view.half(),  # 使用半精度
                            seg_view,
                            size=target_sizes[idx],
                            mode="bilinear",
                            align_corners=False
                        )
                        seg_view.squeeze_(0)  # 恢复原始形状
                        semantic_map = resized_logits[0].argmax(dim=0)
                        semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

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

# 获取颜色映射
color_map = create_ade20k_label_colormap()
print(color_map.shape)  # 应该输出 (151, 3)

image = resize_image(image, 2048)
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

#打印模型推理时的最大显存占用
# 在推理前先清空显存缓存
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()  # 重置显存统计信息
t1 = time.time()
with torch.no_grad():
    # with torch.amp.autocast(device_type="cuda"):
        outputs = model(**inputs)  #
inference_time = time.time() - t1
# 获取最大显存占用（转换为MB）
print(f"模型推理时间: {inference_time:.2f} 秒")
max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"最大显存占用: {max_memory:.2f} MB")





# model predicts class_queries_logits of shape `(batch_size, num_queries)`
# and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

# 将outputs保存为转移到cpu
# outputs_cpu = {k: v.cpu() for k, v in outputs.items()}

t2 = time.time()
# you can pass them to processor for semantic postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[(image.height, image.width)]
)[0]
# predicted_semantic_map = post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])[0]
print(f"后处理时间: {time.time() - t2:.2f} 秒")
max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"最大显存占用: {max_memory:.2f} MB")

# 将 predicted_semantic_map 转换为 PIL 图像

# ADE20k 数据集的颜色映射

t1 = time.time()

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
    # 创建当前类别的二值掩码
    binary_mask = (semantic_map_array == label).astype(np.uint8)
    
    # 使用GenericMask处理掩码
    mask = GenericMask(binary_mask, height, width)
    
    # 填充颜色
    if not mask.has_holes:
        # 对于没有孔洞的掩码，使用多边形填充
        for segment in mask.polygons:
            segment = segment.reshape(-1, 2)
            # 填充区域颜色
            cv2.fillPoly(colored_semantic_map, [segment.astype(np.int32)], color_map[label].tolist())
            # 绘制白色边框
            cv2.polylines(colored_semantic_map, [segment.astype(np.int32)], True, (255, 255, 255), 1)
    else:
        # 对于有孔洞的掩码，直接使用掩码填充
        colored_semantic_map[binary_mask == 1] = color_map[label]
        # 使用findContours检测边缘并绘制白色边框
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(colored_semantic_map, contours, -1, (255, 255, 255), 1)
    
    # 打印当前处理的标签和其面积（用于调试）
    # print(f"处理标签 {label}，面积: {areas[sorted_idxs[list(labels).index(label)]]}")

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

max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"最大显存占用: {max_memory:.2f} MB")



