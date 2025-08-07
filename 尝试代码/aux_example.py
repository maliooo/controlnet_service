from PIL import Image
import requests
from io import BytesIO
from controlnet_aux import HEDdetector, MidasDetector, MLSDdetector, OpenposeDetector, PidiNetDetector, NormalBaeDetector, LineartDetector, LineartAnimeDetector, CannyDetector, ContentShuffleDetector, ZoeDetector, MediapipeFaceDetector, SamDetector, LeresDetector, DWposeDetector
from rich import print
from pathlib import Path
OUTPUT_DIR = Path(__file__).parent / "output"

# load image
url = "https://image6.znzmo.com/1743039310115_8333.png?x-oss-process=image/auto-orient,1/resize,m_fill,w_264,h_198,limit_0"

response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert("RGB").resize((512, 512))

# load checkpoints
# hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
# midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
# mlsd = MLSDdetector.from_pretrained("lllyasviel/Annotators")
# open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
# pidi = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
# normal_bae = NormalBaeDetector.from_pretrained("lllyasviel/Annotators")
# lineart = LineartDetector.from_pretrained("lllyasviel/Annotators")
# lineart_anime = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
# zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
# sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
# mobile_sam = SamDetector.from_pretrained("dhkim2810/MobileSAM", model_type="vit_t", filename="mobile_sam.pt")
# leres = LeresDetector.from_pretrained("lllyasviel/Annotators")
# teed = TEEDdetector.from_pretrained("fal-ai/teed", filename="5_model.pth")
# anyline = AnylineDetector.from_pretrained(
#     "TheMistoAI/MistoLine", filename="MTEED.pth", subfolder="Anyline"
# )

# specify configs, ckpts and device, or it will be downloaded automatically and use cpu by default
# det_config: ./src/controlnet_aux/dwpose/yolox_config/yolox_l_8xb8-300e_coco.py
# det_ckpt: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth
# pose_config: ./src/controlnet_aux/dwpose/dwpose_config/dwpose-l_384x288.py
# pose_ckpt: https://huggingface.co/wanghaofan/dw-ll_ucoco_384/resolve/main/dw-ll_ucoco_384.pth
# import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dwpose = DWposeDetector(det_config=det_config, det_ckpt=det_ckpt, pose_config=pose_config, pose_ckpt=pose_ckpt, device=device)

# instantiate
canny = CannyDetector()
# content = ContentShuffleDetector()
# face_detector = MediapipeFaceDetector()
# lineart_standard = LineartStandardDetector()


# process
# processed_image_hed = hed(img)
# processed_image_midas = midas(img)
# processed_image_mlsd = mlsd(img)
# processed_image_open_pose = open_pose(img, hand_and_face=True)
# processed_image_pidi = pidi(img, safe=True)
# processed_image_normal_bae = normal_bae(img)
# processed_image_lineart = lineart(img, coarse=True)
# processed_image_lineart_anime = lineart_anime(img)
# processed_image_zoe = zoe(img)
# processed_image_sam = sam(img)
# processed_image_leres = leres(img)
# processed_image_teed = teed(img, detect_resolution=1024)
# processed_image_anyline = anyline(img, detect_resolution=1280)

processed_image_canny = canny(img)
print(f"processed_image_canny: {processed_image_canny}")
processed_image_canny.save(OUTPUT_DIR / "processed_image_canny.png")
# processed_image_content = content(img)
# processed_image_mediapipe_face = face_detector(img)
# processed_image_dwpose = dwpose(img)
# processed_image_lineart_standard = lineart_standard(img, detect_resolution=1024)