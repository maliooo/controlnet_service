from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import torch
import os
import numpy as np
import pandas as pd
import time
import cv2
from typing import Optional, List, Tuple, Union, Dict
from pathlib import Path


class OneFormerSegmentation:
    """OneFormer语义分割模型封装类"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化OneFormer分割模型
        
        参数:
            model_path: 模型路径，默认使用ADE20K的Swin-Large模型
            device: 运行设备，可以是'cuda'或'cpu'，默认自动检测
        """
        # 检查是否有可用的GPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 默认模型路径
        if model_path is None:
            model_path = r"models/oneformer_ade20k_swin_large"
        
        # 加载模型和处理器
        self.processor = OneFormerProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(model_path, local_files_only=True)
        self.model = self.model.to(self.device)
        
        # 加载ADE20K颜色映射
        self.color_map, self.color_map_label_to_name = self._create_ade20k_label_colormap()
        self.ade20k_csv_path = None
        
        # 性能统计
        # self.stats = {}
    
    def _create_ade20k_label_colormap(self):
        """创建ADE20K分割基准中使用的标签颜色映射"""
        # 初始化颜色映射数组，ADE20k有150个类别（不包括背景类0）
        colormap = np.zeros((150, 3), dtype=np.uint8)
        colormap_label_to_name = {}
        
        try:
            # 读取CSV文件
            csv_path = Path(__file__).parent / "csv" / "ade20k.csv"
            self.ade20k_csv_path = csv_path
            df = pd.read_csv(csv_path)
            print(f"df.columns: {df.columns}")
            
            # 解析颜色代码并填充colormap
            for idx, row in df.iterrows():
                color_str = row['Color_Code (R,G,B)']
                # 提取RGB值
                rgb = eval(color_str)  # 将字符串"(R,G,B)"转换为元组
                colormap[idx] = np.array(rgb)
                colormap_label_to_name[idx] = row['Name']
        except Exception as e:
            print(f"加载颜色映射失败: {e}")
            print("使用随机颜色映射")
            # 如果CSV加载失败，使用随机颜色
            np.random.seed(42)
            for i in range(150):
                colormap[i] = np.random.randint(0, 256, 3)
        
        return colormap, colormap_label_to_name
    
    def resize_image(self, image: Image.Image, size: int) -> Image.Image:
        """将图片按照短边缩放到指定大小，长边等比例缩放"""
        width, height = image.size
        if width < height:
            new_width = size
            new_height = int(height * size / width)
        else:
            new_height = size
            new_width = int(width * size / height)
        return image.resize((new_width, new_height))
    
    def segment(self, 
                image_path: str = None, 
                image: Image.Image = None, 
                resize_to: int = 512, 
                draw_contours: bool = True,
                min_area_threshold: int = 50,
                return_combined: bool = False) -> Dict:
        """
        对图像进行语义分割
        
        参数:
            image_path: 图像路径
            image: PIL图像对象，如果提供则优先使用
            resize_to: 将图像短边缩放到的大小，长边等比例缩放
            draw_contours: 是否在分割结果上绘制轮廓
            min_area_threshold: 绘制轮廓的最小区域面积阈值
            return_combined: 是否返回原图和分割结果的组合图像
            
        返回:
            包含分割结果和性能统计的字典
        """
        t_start = time.time()
        stats = {}
        
        # 加载图像
        if image is None and image_path is not None:
            image = Image.open(image_path)
        elif image is None and image_path is None:
            raise ValueError("必须提供image_path或image参数")
        
        # 调整图像大小
        if resize_to:
            image = self.resize_image(image, resize_to)
        
        print(f"处理图像大小: {image.size}")
        
        # 准备输入
        t_preprocess = time.time()
        inputs = self.processor(image, ["semantic"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        stats['预处理时间'] = time.time() - t_preprocess
        
        # 清空显存缓存并重置统计信息
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 模型推理
        t_inference = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        stats['推理时间'] = time.time() - t_inference
        
        # 后处理
        t_postprocess = time.time()
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        stats['后处理时间'] = time.time() - t_postprocess
        
        # 可视化
        t_visualization = time.time()
        vis_result = self._visualize_segmentation(
            predicted_semantic_map, 
            draw_contours=draw_contours,
            min_area_threshold=min_area_threshold
        )
        colored_map = vis_result['semantic_map_image']
        label_info = vis_result['label_info']
        vis_stats = vis_result['stats']
        # 合并stats
        stats = {**stats, **vis_stats}

        stats['可视化时间'] = time.time() - t_visualization
        
        # 创建组合图像
        combined_image = None
        if return_combined:
            combined_image = Image.new('RGB', (image.width * 2, image.height))
            combined_image.paste(image, (0, 0))
            combined_image.paste(colored_map, (image.width, 0))
        
        stats['总处理时间'] = time.time() - t_start
        stats['最大显存占用(MB)'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return {
            'original_image': image,
            'segmentation_map': predicted_semantic_map,
            'colored_segmentation': colored_map,
            'combined_image': combined_image,
            'label_info': label_info,
            'stats': stats
        }
    
    def _visualize_segmentation(self, 
                               semantic_map_array: torch.Tensor, 
                               draw_contours: bool = True,
                               min_area_threshold: int = 50) -> Tuple[Image.Image, Dict]:
        """
        将分割结果可视化为彩色图像
        
        参数:
            semantic_map_array: 分割结果张量
            draw_contours: 是否绘制轮廓
            min_area_threshold: 绘制轮廓的最小区域面积阈值
            
        返回:
            彩色分割图像和标签信息
        """
        t_vis_start = time.time()
        stats = {}
        # 创建彩色分割图（在GPU上）
        height, width = semantic_map_array.shape
        colored_semantic_map = torch.zeros((height, width, 3), dtype=torch.uint8, device=self.device)
        
        # 获取唯一的标签和它们的面积计数
        labels, areas = torch.unique(semantic_map_array, return_counts=True)
        print(f"labels: {labels}")
        label_to_name = [self.color_map_label_to_name[label.item()] for label in labels]
        print(f"label_to_name: {label_to_name}")

        # 按面积从大到小排序
        sorted_idxs = torch.argsort(areas, descending=True)
        labels = labels[sorted_idxs]
        areas = areas[sorted_idxs]
        
        # 将color_map转换为GPU张量
        color_map_gpu = torch.from_numpy(self.color_map).to(self.device)
        
        # 批量处理所有标签
        valid_labels = 0
        label_info = {}
        
        for i, label in enumerate(labels):
            if label >= len(self.color_map):
                continue
                
            # 创建当前类别的二值掩码
            mask = (semantic_map_array == label)
            print(f"{label}, mask: {mask.shape}")
            
            # 直接在GPU上进行颜色填充
            colored_semantic_map[mask] = color_map_gpu[label]
            
            # 记录标签信息
            label_info[label.item()] = {
                'area': areas[i].item(),
                'percentage': (areas[i].item() / (height * width) * 100),
                'color': self.color_map[label.item()].tolist(),
                "name": self.color_map_label_to_name[label.item()],
                "mask": mask.cpu().numpy()
            }
            
            valid_labels += 1
        
        # 将结果转移到CPU
        colored_semantic_map = colored_semantic_map.cpu().numpy()
        
        # 绘制轮廓
        if draw_contours:
            # 将semantic_map_array转移到CPU
            semantic_map_cpu = semantic_map_array.cpu().numpy()
            
            # 创建一个边框掩码层
            contour_mask = np.zeros((height, width), dtype=np.uint8)
            contour_count = 0
            
            for label in labels[labels < len(self.color_map)].cpu().numpy():
                # 如果该标签的面积太小，跳过边框绘制
                label_area = areas[torch.where(labels == label)[0]].item()
                if label_area < min_area_threshold:
                    continue
                    
                binary_mask = (semantic_map_cpu == label).astype(np.uint8)
                
                # 使用更高效的轮廓检测参数
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 只绘制轮廓到掩码上
                cv2.drawContours(contour_mask, contours, -1, 1, 1)
                contour_count += len(contours)
            
            # 一次性将所有轮廓应用到彩色图像上
            colored_semantic_map[contour_mask == 1] = [255, 255, 255]
            
            stats['轮廓数量'] = contour_count
        
        # 转换为PIL图像
        semantic_map_image = Image.fromarray(colored_semantic_map)
        
        stats['可视化处理时间'] = time.time() - t_vis_start
        stats['有效标签数量'] = valid_labels
        
        return {
            'semantic_map_image': semantic_map_image,
            'label_info': label_info,
            'stats': stats
        }
    
    def save_results(self, 
                    results: Dict, 
                    output_dir: str = None, 
                    base_filename: str = "segmentation_result"):
        """
        保存分割结果
        
        参数:
            results: segment方法返回的结果字典
            output_dir: 输出目录，默认为当前目录
            base_filename: 基础文件名
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "output"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存彩色分割图
        colored_path = output_dir / f"{base_filename}.png"
        results['colored_segmentation'].save(colored_path)
        print(f"分割结果已保存到: {colored_path}")
        
        # 保存组合图像
        if results['combined_image'] is not None:
            combined_path = output_dir / f"{base_filename}_combined.png"
            results['combined_image'].save(combined_path)
            print(f"组合图像已保存到: {combined_path}")
        
        # 保存标签信息
        label_info_path = output_dir / f"{base_filename}_labels.csv"
        label_df = pd.DataFrame.from_dict(results['label_info'], orient='index')
        label_df.index.name = 'label_id'
        label_df.to_csv(label_info_path)
        print(f"标签信息已保存到: {label_info_path}")
        
        # 打印性能统计
        print("\n性能统计:")
        for key, value in results['stats'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


# 示例用法
if __name__ == "__main__":
    OUTPUT_DIR = Path(__file__).parent / "output"
    # 初始化分割器
    segmenter = OneFormerSegmentation()
    
    # 进行分割
    image_path = "images/bedroom.jpg"
    results = segmenter.segment(
        # image_path=image_path,
        image=Image.open(image_path),
        resize_to=512,
        draw_contours=True,
        return_combined=True
    )
    
    # 保存结果
    segmenter.save_results(results, output_dir=OUTPUT_DIR)

    for label_index, label_info in results['label_info'].items():
        print(f"label_index: {label_index}, label_info: {label_info}")
        mask = label_info['mask']
        name = label_info['name']
        # numpy 转 PIL
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.convert('RGB')
        mask_pil.save(OUTPUT_DIR / f"mask_{name}.png")
