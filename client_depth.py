import requests
import base64
from PIL import Image
import io
from pathlib import Path
import time
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="深度估计客户端")
    parser.add_argument("--short_size", type=int, default=512,
                      help="图片短边缩放后的大小")
    parser.add_argument("--model_name", type=str, default="large", 
                      choices=["large", "indoor", "outdoor"],
                      help="模型名称: large, indoor, outdoor")
    parser.add_argument("--server_url", type=str, default="http://localhost:28003",
                      help="服务器地址")
    parser.add_argument("--num_requests", type=int, default=1,
                      help="请求次数")
    parser.add_argument("--request_type", type=str, default="base64", 
                      choices=["file", "base64", "url"],
                      help="请求类型: file, base64, url")
    parser.add_argument("--image_path", type=str, default=None,
                      help="图片路径（用于file和base64模式）")
    parser.add_argument("--image_url", type=str, 
                      default="https://image6.znzmo.com/cover/264x/0_291e22dc52c1133741c68fbff635e6bc6b1ef721.png",
                      help="图片URL（用于url模式）")
    return parser.parse_args()

def request_with_file(image_path, server_url, model_name, short_size):
    """使用文件方式请求"""
    with open(image_path, "rb") as f:
        files = {"file": f}
        data = {
            "model_name": model_name,
            "short_size": str(short_size)  # 转换为字符串
        }
        response = requests.post(
            f"{server_url}/estimate_depth",
            files=files,
            data=data
        )
    return response

def request_with_base64(image_path, server_url, model_name, short_size):
    """使用base64方式请求"""
    with open(image_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode()
        data = {
            "image_url_or_base64": image_base64,
            "model_name": model_name,
            "short_size": short_size  # JSON请求中可以是整数
        }
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{server_url}/estimate_depth_base64",
            json=data,
            headers=headers
        )
    return response

def request_with_url(image_url, server_url, model_name, short_size):
    """使用URL方式请求"""
    data = {
        "image_url_or_base64": image_url,
        "model_name": model_name,
        "short_size": short_size  # JSON请求中可以是整数
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"{server_url}/estimate_depth_url",
        json=data,
        headers=headers
    )
    return response

def save_result(result, output_path, index, model_name):
    """保存结果图片"""
    if "error" in result:
        print(f"处理出错: {result['error']}")
        return
    
    if "depth_image" not in result:
        print(f"响应格式错误，缺少 depth_image 字段")
        print(f"响应内容: {result}")
        return
    
    depth_image_b64 = result["depth_image"]
    depth_image_data = base64.b64decode(depth_image_b64)
    depth_image = Image.open(io.BytesIO(depth_image_data))
    
    save_path = output_path / f"depth_result_{model_name}_{index}.png"
    depth_image.save(save_path)
    print(f"深度图已保存: {save_path}")
    print(f"深度图尺寸: {depth_image.size}")

def main():
    args = parse_args()
    
    # 设置输出目录
    output_path = Path(__file__).parent / "results_depth"
    output_path.mkdir(exist_ok=True)
    
    # 根据请求类型选择图片路径或URL
    if args.request_type in ["file", "base64"]:
        if args.image_path is None:
            args.image_path = "./images/bedroom.jpg"
        if not os.path.exists(args.image_path):
            raise FileNotFoundError(f"图片文件不存在: {args.image_path}")
    elif args.request_type == "url":
        if args.image_url is None:
            raise ValueError("URL模式需要提供 --image_url 参数")
    
    print(f"请求参数为：model_name={args.model_name}, short_size={args.short_size}")
    print(f"请求类型：{args.request_type}")
    
    for i in range(args.num_requests):
        try:
            t_start = time.time()
            
            # 根据请求类型选择不同的请求方式
            if args.request_type == "file":
                response = request_with_file(args.image_path, args.server_url, args.model_name, args.short_size)
            elif args.request_type == "base64":
                response = request_with_base64(args.image_path, args.server_url, args.model_name, args.short_size)
            else:  # url
                response = request_with_url(args.image_url, args.server_url, args.model_name, args.short_size)
            
            # 检查响应状态码
            response.raise_for_status()
            
            # 获取结果
            result = response.json()
            # FastAPI的HTTPException会返回包含detail的JSON
            if "detail" in result:
                print(f"第{i+1}次请求处理异常: {result['detail']}")
                continue
                
            save_result(result, output_path, i, args.model_name)
            print(f"第{i+1}次请求处理时间为: {(time.time() - t_start):.2f}秒")
            
        except requests.exceptions.RequestException as e:
            print(f"第{i+1}次请求网络错误: {str(e)}")
        except Exception as e:
            print(f"第{i+1}次请求处理异常: {str(e)}")
            print(f"异常类型: {type(e).__name__}")
            import traceback
            print(f"异常堆栈: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 