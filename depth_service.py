# https://huggingface.co/docs/transformers/main/model_doc/depth_anything_v2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image, ImageOps
import os
import requests
from pathlib import Path
import time
import gc
import argparse
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException
import uvicorn
import io
import base64
import asyncio
from pydantic import BaseModel
from typing import Optional
from rich import print
import hashlib
from collections import OrderedDict

# 创建一个信号量来控制GPU推理的并发数
gpu_semaphore = asyncio.Semaphore(1)  # 设置为1表示同一时间只允许一个GPU推理任务

class ModelImageCache:
    def __init__(self, max_size=200):
        self.max_size = max_size
        # 为每个模型创建独立的缓存
        self.model_caches = {
            "large": OrderedDict(),
            "indoor": OrderedDict(),
            "outdoor": OrderedDict()
        }
    
    def get_image_hash(self, image):
        """计算图片的哈希值"""
        if isinstance(image, str):
            # 如果是base64字符串
            image_data = base64.b64decode(image)
        elif isinstance(image, Image.Image):
            # 如果是PIL Image对象
            image_data = image.tobytes()
        else:
            # 如果是文件内容
            image_data = image
        return hashlib.md5(image_data).hexdigest()
    
    def get(self, image, model_name):
        """获取缓存的结果"""
        key = self.get_image_hash(image)
        cache = self.model_caches[model_name]
        
        if key in cache:
            # 将访问的项移到最新
            value = cache.pop(key)
            cache[key] = value
            return value
        return None
    
    def put(self, image, result, model_name):
        """将结果存入缓存"""
        key = self.get_image_hash(image)
        cache = self.model_caches[model_name]
        
        if key in cache:
            # 如果已存在，先删除
            cache.pop(key)
        elif len(cache) >= self.max_size:
            # 如果缓存已满，删除最旧的项
            cache.popitem(last=False)
        cache[key] = result

# 创建全局缓存实例
image_cache = ModelImageCache(max_size=500)

# 定义请求数据模型
class ImageRequestData(BaseModel):
    image_url_or_base64: Optional[str] = None
    model_name: Optional[str] = "large"  # 可以是 large, indoor, outdoor
    short_size: Optional[int] = 512

# 依赖函数，用于解析表单数据
def get_image_request_data(
    image_url_or_base64: Optional[str] = Form(None),
    model_name: Optional[str] = Form("large"),
    short_size: Optional[str] = Form(512),
) -> ImageRequestData:
    return ImageRequestData(
        image_url_or_base64 = image_url_or_base64,
        model_name = model_name,
        short_size = short_size
    )


class DepthEstimator:
    def __init__(self, device=None, use_fp16=True, model_dir=None):
        """
        初始化深度估计器
        Args:
            model_path: 模型路径
            device: 设备，如果为None则自动选择
            use_fp16: 是否使用FP16精度
        """

        # 1. 设置设备
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16
        print(f"使用设备: {self.device}")

        # 2. 加载模型, 这里有3种模型，分别是 large, indoor, outdoor
        if model_dir is None:
            model_large_path = str(Path(__file__).parent / "Depth-Anything-V2-Large-hf")
        else:
            model_large_path = str(Path(model_dir)/ "Depth-Anything-V2-Large-hf")
        self.image_processor_large = AutoImageProcessor.from_pretrained(model_large_path)
        self.model_large = AutoModelForDepthEstimation.from_pretrained(model_large_path)

        if model_dir is None:
            model_indoor_path = str(Path(__file__).parent / "Depth-Anything-V2-Metric-Indoor-Large-hf")
        else:
            model_indoor_path = str(Path(model_dir) / "Depth-Anything-V2-Metric-Indoor-Large-hf")
        self.image_processor_indoor = AutoImageProcessor.from_pretrained(model_indoor_path)
        self.model_indoor = AutoModelForDepthEstimation.from_pretrained(model_indoor_path)

        if model_dir is None:
            model_outdoor_path = str(Path(__file__).parent / "Depth-Anything-V2-Metric-Outdoor-Large-hf")
        else:
            model_outdoor_path = str(Path(model_dir) / "Depth-Anything-V2-Metric-Outdoor-Large-hf")
        self.image_processor_outdoor = AutoImageProcessor.from_pretrained(model_outdoor_path)
        self.model_outdoor = AutoModelForDepthEstimation.from_pretrained(model_outdoor_path)
        
        # 3. 将模型移动到指定设备
        self.model_large = self.model_large.to(self.device)
        if self.use_fp16:
            self.model_large = self.model_large.half()

        self.model_indoor = self.model_indoor.to(self.device)
        if self.use_fp16:
            self.model_indoor = self.model_indoor.half()

        self.model_outdoor = self.model_outdoor.to(self.device)
        if self.use_fp16:
            self.model_outdoor = self.model_outdoor.half()
        
        # 4. 将模型和图像处理器打包成一个字典
        self.name_2_model = {
            "large": (self.image_processor_large, self.model_large),
            "indoor" : (self.image_processor_indoor, self.model_indoor),
            "outdoor": (self.image_processor_outdoor, self.model_outdoor)
        }

        # 5. 打印当前GPU显存使用情况
        print("depth-anything-v2模型加载完成，当前显存使用：")
        self.print_gpu_memory()
    
    def print_gpu_memory(self):
        """打印当前GPU显存使用情况"""
        if torch.cuda.is_available():
            print(f"GPU显存使用: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            print(f"GPU显存缓存: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")
    
    def resize_image(self, image, short_size=512):
        """将图片按照短边缩放到指定大小"""
        width, height = image.size
        if width < height:
            new_width = short_size
            new_height = int(height * short_size / width)
        else:
            new_height = short_size
            new_width = int(width * short_size / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"resize_image 处理图片完成，size大小： {image.size}")
        return image
    
    def estimate_depth(self, image, model_name="large"):
        """
        估计图片深度
        Args:
            image: PIL Image对象 (已经缩放过的)
            model_name: 模型名称
        Returns:
            depth_img: 深度图PIL Image对象
        """

        image_processor, depth_model = self.name_2_model[model_name]
        
        # 准备输入
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.use_fp16:
            inputs = {k: v.half() for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = depth_model(**inputs)
        
        # 后处理
        post_processed_output = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        
        predicted_depth = post_processed_output[0]["predicted_depth"]
        depth = (predicted_depth - predicted_depth.min()) / (predicted_depth.max() - predicted_depth.min())
        depth = depth.detach().cpu().numpy() * 255
        if model_name in ["indoor", "outdoor"]:
            depth = depth * (-1) + 255 # 如果是 indoor 和 outdoor模型，黑白反转
        depth = depth.astype("uint8")
        
        # 转换为PIL图像
        depth_img = Image.fromarray(depth)
        depth_img = depth_img.convert("RGB")
        
        # 清理显存
        del outputs, predicted_depth, depth
        torch.cuda.empty_cache()
        gc.collect()
        
        return depth_img

def create_app(estimator):
    """创建FastAPI应用"""
    app = FastAPI(title="深度估计服务")
    
    @app.post("/estimate_depth")
    async def estimate_depth(
        file: UploadFile = File(...),
        request_data: ImageRequestData = Depends(get_image_request_data)
    ):
        
        # 1. 读取上传的图片
        print(f"[yellow]调用函数 estimate_depth， 请求参数为: {request_data}[/yellow]")
        contents = await file.read()
        
        # 2. 调整图片大小
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
        image = estimator.resize_image(image, request_data.short_size)
        
        # 3. 检查缓存（在缩放后检查）
        cached_result = image_cache.get(image.tobytes(), request_data.model_name)
        if cached_result:
            print("从缓存中获取结果")
            return cached_result
        
        # 4. 如果缓存中没有结果，则进行推理
        t_start = time.time()
        
        # 使用信号量控制GPU推理
        async with gpu_semaphore:
            # 估计深度
            depth_img = estimator.estimate_depth(
                image,
                model_name=request_data.model_name
            )
            
            # 将结果转换为base64
            buffered = io.BytesIO()
            depth_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            result = {"depth_image": img_str}
            
            # 存入缓存（使用缩放后的图片）
            image_cache.put(image.tobytes(), result, request_data.model_name)
            
            print(f"服务处理时间为: {(time.time() - t_start):.2f}")
        return result

    @app.post("/estimate_depth_base64")
    async def estimate_depth_base64(
        request_data: ImageRequestData
    ):
        try:
            # 1. 解码base64图片, 转为RGB图片, 缩放到指定大小
            print(f"[yellow]调用函数 estimate_depth_base64， 请求参数为: {request_data.model_name}, {request_data.short_size}[/yellow]")
            image_data = base64.b64decode(request_data.image_url_or_base64)
            image = Image.open(io.BytesIO(image_data))
            image = image.convert("RGB")
            image = estimator.resize_image(image, request_data.short_size)
            print(f"base64服务处理图片大小： {image.size}")
            
            # 2. 检查缓存（在缩放后检查）
            cached_result = image_cache.get(image.tobytes(), request_data.model_name)
            if cached_result:
                print("从缓存中获取结果")
                return cached_result
            
            # 3. 如果缓存中没有结果，则进行推理
            t_start = time.time()
            
            # 使用信号量控制GPU推理
            async with gpu_semaphore:
                # 估计深度
                depth_img = estimator.estimate_depth(
                    image,
                    model_name=request_data.model_name
                )
                
                # 将结果转换为base64
                buffered = io.BytesIO()
                depth_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                result = {"depth_image": img_str}
                
                # 存入缓存（使用缩放后的图片）
                image_cache.put(image.tobytes(), result, request_data.model_name)
                
                print(f"服务处理时间为: {(time.time() - t_start):.2f}")
            return result
        except Exception as e:
            return {"error": f"处理base64图片时出错: {str(e)}"}

    @app.post("/estimate_depth_url")
    async def estimate_depth_url(
        request_data: ImageRequestData
    ):
        try:
            print(f"[yellow]调用函数 estimate_depth_url， 请求参数为: {request_data}[/yellow]")
            # 1. 下载图片
            response = requests.get(request_data.image_url_or_base64)
            response.raise_for_status()
            image_content = response.content
            
            # 2. 将图片转为RGB图片, 缩放到指定大小
            image = Image.open(io.BytesIO(image_content))
            image = image.convert("RGB")
            image = estimator.resize_image(image, request_data.short_size)
            print(f"url服务处理图片大小： {image.size}")
            
            # 3. 检查缓存（在缩放后检查）
            cached_result = image_cache.get(image.tobytes(), request_data.model_name)
            if cached_result:
                print("从缓存中获取结果")
                return cached_result
            
            # 4. 如果缓存中没有结果，则进行推理
            t_start = time.time()
            
            # 5. 使用信号量控制GPU推理
            async with gpu_semaphore:
                # 估计深度
                depth_img = estimator.estimate_depth(
                    image,
                    model_name=request_data.model_name
                )
                
                # 将结果转换为base64
                buffered = io.BytesIO()
                depth_img.save(buffered, format="PNG")
                print(f"url服务处理图片大小： {depth_img.size}")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                result = {"depth_image": img_str}
                
                # 存入缓存（使用缩放后的图片）
                image_cache.put(image.tobytes(), result, request_data.model_name)
                
                print(f"服务处理时间为: {(time.time() - t_start):.2f}")
            return result
        except Exception as e:
            print(f"[red]处理URL图片时出错: {str(e)}[/red]")
            return {"error": f"处理URL图片时出错: {str(e)}"}
    
    return app

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="深度估计服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8002, help="服务端口")
    parser.add_argument("--device", type=str, default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--no_fp16", action="store_true", help="不使用FP16精度")
    parser.add_argument("--max_concurrent", type=int, default=1, help="最大并发GPU推理数量")
    parser.add_argument("--model_dir", type=str, default="models/Depth-Anything-V2", help="模型目录路径")
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 设置并发数
    global gpu_semaphore
    gpu_semaphore = asyncio.Semaphore(args.max_concurrent)
    
    # 初始化估计器
    estimator = DepthEstimator(
        device=args.device,
        use_fp16=not args.no_fp16,
        model_dir=args.model_dir
    )
    
    # 创建FastAPI应用
    app = create_app(estimator)
    
    # 启动服务
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()