from src.comfyui_controlnet_aux.src.custom_controlnet_aux.oneformer import get_oneformer_metadata
from detectron2.data import MetadataCatalog

metadata = MetadataCatalog.get("ade20k_sem_seg_val")
# metadata.stuff_colors = self.color_map.tolist()
metadata = get_oneformer_metadata(None, "250_16_swin_l_oneformer_ade20k_160k.pth", metadata=metadata)
print(metadata)