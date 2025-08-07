from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
from PIL import Image
import requests
url = "https://huggingface.co/datasets/shi-labs/oneformer_demo/blob/main/ade20k.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

# Loading a single model for all three tasks
processor = OneFormerProcessor.from_pretrained("models/oneformer_ade20k_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained("models/oneformer_ade20k_swin_large")

# Semantic Segmentation
semantic_inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
semantic_outputs = model(**semantic_inputs)
# pass through image_processor for postprocessing
predicted_semantic_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

# Instance Segmentation
instance_inputs = processor(images=image, task_inputs=["instance"], return_tensors="pt")
instance_outputs = model(**instance_inputs)
# pass through image_processor for postprocessing
predicted_instance_map = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]

# Panoptic Segmentation
panoptic_inputs = processor(images=image, task_inputs=["panoptic"], return_tensors="pt")
panoptic_outputs = model(**panoptic_inputs)
# pass through image_processor for postprocessing
predicted_semantic_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]["segmentation"]
