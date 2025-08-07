export HF_ENDPOINT=https://hf-mirror.com
nohup /root/miniconda3/envs/aux/bin/python  depth_service.py --host 0.0.0.0 --port 28003  > depth.log 2>&1 &