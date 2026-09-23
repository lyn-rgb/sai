import os
import numpy as np
from PIL import Image
import cv2
from insightface.app import FaceAnalysis


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_kps(image, faces):
    dimg = image.copy()
    for i in range(len(faces)):
        face = faces[i]
        box = face.bbox.astype(np.int32)
        color = (0, 0, 255)
        cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
        if face.kps is not None:
            kps = face.kps.astype(np.int32)
            #print(landmark.shape)
            for l in range(kps.shape[0]):
                color = (0, 0, 255)
                if l == 0 or l == 3:
                    color = (0, 255, 0)
                cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                            2)
        if face.gender is not None and face.age is not None:
            cv2.putText(dimg,'%s,%d'%(face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
    return dimg


def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":
    
    image_path = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/example_prompts/pngs/girl-001-small.jpg"
    # Load face encoder
    app = FaceAnalysis(name='antelopev2', root='/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/ckpts/InsightFace', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    face_image = Image.open(image_path).resize((512, 512), resample=Image.BILINEAR)
    face_image_cv = convert_from_image_to_cv2(face_image)
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image_cv, [face_info])
    cv2.imwrite("/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/CloneMyFaceCloneMyVoice/sai/debug/face_kps.jpg", face_kps)
    print(f"face embedding shape: {face_emb.shape}, mean: {np.mean(face_emb)}, std: {np.std(face_emb)}")

