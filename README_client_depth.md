# 深度估计客户端使用说明

## 概述

`client_depth.py` 是一个用于测试深度估计服务的客户端程序，支持多种请求方式和参数配置。

## 功能特性

- 支持三种请求方式：文件上传、base64编码、URL
- 支持三种模型：large、indoor、outdoor
- 可配置图片缩放尺寸
- 支持批量请求测试
- 自动保存结果图片

## 安装依赖

```bash
pip install requests pillow
```

## 使用方法

### 基本用法

```bash
# 使用默认参数（base64请求，large模型，512尺寸）
python client_depth.py

# 指定图片路径
python client_depth.py --image_path ./images/test.jpg

# 使用URL请求
python client_depth.py --request_type url --image_url "https://example.com/image.jpg"
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--short_size` | int | 512 | 图片短边缩放后的大小 |
| `--model_name` | str | large | 模型名称（large/indoor/outdoor） |
| `--server_url` | str | http://localhost:8002 | 服务器地址 |
| `--num_requests` | int | 1 | 请求次数 |
| `--request_type` | str | base64 | 请求类型（file/base64/url） |
| `--image_path` | str | None | 图片路径（用于file和base64模式） |
| `--image_url` | str | 默认URL | 图片URL（用于url模式） |

### 使用示例

#### 1. 文件上传模式

```bash
python client_depth.py \
    --request_type file \
    --image_path ./images/bedroom.jpg \
    --model_name large \
    --short_size 512
```

#### 2. Base64编码模式

```bash
python client_depth.py \
    --request_type base64 \
    --image_path ./images/bedroom.jpg \
    --model_name indoor \
    --short_size 256
```

#### 3. URL模式

```bash
python client_depth.py \
    --request_type url \
    --image_url "https://example.com/image.jpg" \
    --model_name outdoor \
    --short_size 1024
```

#### 4. 批量测试

```bash
# 进行10次请求测试
python client_depth.py \
    --request_type base64 \
    --num_requests 10 \
    --model_name large
```

## 输出结果

- 结果图片保存在 `results_depth/` 目录下
- 文件名格式：`depth_result_{model_name}_{index}.png`
- 控制台会显示处理时间和保存路径

## 错误处理

客户端包含完善的错误处理机制：

- 网络连接错误
- 文件不存在错误
- 服务端错误响应
- 参数验证错误

## 测试脚本

使用 `test_client_depth.py` 可以自动测试客户端的各种功能：

```bash
python test_client_depth.py
```

## 注意事项

1. 确保深度估计服务已经启动并运行在指定端口（默认8002）
2. 图片文件路径需要正确设置
3. 网络请求需要确保网络连接正常
4. 大量请求时注意服务端的并发限制

## 故障排除

### 常见问题

1. **连接被拒绝**
   - 检查服务是否已启动
   - 确认端口号是否正确

2. **文件不存在**
   - 检查图片路径是否正确
   - 确认文件是否存在

3. **模型加载失败**
   - 检查服务端模型文件是否存在
   - 确认模型名称是否正确

4. **内存不足**
   - 减少并发请求数量
   - 降低图片尺寸

## 开发说明

客户端代码结构清晰，易于扩展：

- `parse_args()`: 参数解析
- `request_with_*()`: 不同请求方式的实现
- `save_result()`: 结果保存
- `main()`: 主程序逻辑

可以根据需要添加新的功能或修改现有功能。 