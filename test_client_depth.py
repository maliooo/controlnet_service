#!/usr/bin/env python3
"""
深度估计客户端测试脚本
"""

import subprocess
import sys
import time
from pathlib import Path

def test_client():
    """测试深度估计客户端"""
    print("开始测试深度估计客户端...")
    
    # 检查客户端文件是否存在
    client_path = Path("client_depth.py")
    if not client_path.exists():
        print("错误：client_depth.py 文件不存在")
        return False
    
    # 测试参数
    test_cases = [
        {
            "name": "测试base64请求",
            "args": ["python", "client_depth.py", "--request_type", "base64", "--num_requests", "1"]
        },
        {
            "name": "测试URL请求", 
            "args": ["python", "client_depth.py", "--request_type", "url", "--num_requests", "1"]
        },
        {
            "name": "测试不同模型",
            "args": ["python", "client_depth.py", "--request_type", "base64", "--model_name", "indoor", "--num_requests", "1"]
        },
        {
            "name": "测试不同尺寸",
            "args": ["python", "client_depth.py", "--request_type", "base64", "--short_size", "256", "--num_requests", "1"]
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{test_case['name']}...")
        try:
            result = subprocess.run(
                test_case['args'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                print(f"✅ {test_case['name']} 成功")
                if result.stdout:
                    print("输出:", result.stdout.strip())
            else:
                print(f"❌ {test_case['name']} 失败")
                print("错误输出:", result.stderr.strip())
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_case['name']} 超时")
        except Exception as e:
            print(f"❌ {test_case['name']} 异常: {str(e)}")
    
    print("\n测试完成！")
    return True

if __name__ == "__main__":
    test_client() 