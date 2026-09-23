from PIL import Image
import numpy as np

import torch
from face_cropper.cropper import Cropper
from face_cropper.crop_config import CropConfig


def crop_image(cropper, images: torch.Tensor):
    # image, tensor, [-1, 1], BxCxHxW, to [0, 255], RGB numpy
    _images = images.permute(0, 2, 3, 1)  # BxHxWxC
    _images = ((_images + 1) / 2) * 255    # [0, 255]
    _images = _images.cpu().numpy()
    cropped_images = []
    try:
        for image in _images:
            cropped_image = cropper.crop_source_image(image, cropper.crop_cfg)["img_crop"]
            #print(f"cropped_image shape: {cropped_image.shape}, value range: [{cropped_image.min()}, {cropped_image.max()}]")
            cropped_image = torch.from_numpy(cropped_image).float()
            cropped_image = (cropped_image.permute(2, 0, 1) / 255) * 2 - 1
            cropped_images.append(cropped_image)
        cropped_images = torch.stack(cropped_images)
    except Exception as e:
        print(f"Error <{e}> occurrs during crop images.")
        cropped_images = images
    return cropped_images    # [-1, 1], BxCxHxW
    
    
if __name__ == "__main__":
    image_path = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/logs/sai_1M_baseline/debug/ip_image.jpg"
    save_path = "/inspire/hdd/project/agileapplication/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/logs/sai_1M_baseline/debug/cropped_ip_image.jpg"
    image = Image.open(image_path).convert('RGB')
    image = (np.array(image, np.float32) / 255.) * 2 - 1 # [0, 255] -> [-1, 1]
    images = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2)
    cropper: Cropper = Cropper(crop_cfg=CropConfig, device_id=torch.cuda.current_device())
    # image or video, [-1, 1], BxCxHxW
    cropped_images = crop_image(cropper, images)
    cropped_image = cropped_images.permute(0, 2, 3, 1)[0]
    cropped_image = ((cropped_image + 1) / 2) * 255.
    cropped_image = cropped_image.cpu().numpy().astype(np.uint8)
    print(cropped_image.shape)
    cropped_image = Image.fromarray(cropped_image)
    cropped_image.save(save_path)
    print(f"Cropped image has been saved into {save_path}")
    
    
    
    
    
    
    