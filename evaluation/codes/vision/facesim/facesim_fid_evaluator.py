import os
import gc
import sys

import cv2
import numpy as np
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PIL import Image
from torchvision import models, transforms


current_file_path = os.path.abspath(__file__)
facesim_root = os.path.dirname(current_file_path)
if facesim_root not in sys.path:
    sys.path.insert(0, facesim_root)

from curricularface import get_model


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()
    
    
def load_image(image):
    img = image.convert('RGB')
    img = transforms.Resize((299, 299))(img)  # Resize to Inception input size
    img = transforms.ToTensor()(img)
    return img.unsqueeze(0)  # Add batch dimension


def matrix_sqrt(matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=0))
    sqrt_matrix = (eigenvectors * sqrt_eigenvalues).mm(eigenvectors.T)
    return sqrt_matrix


def pad_np_bgr_image(np_image, scale=1.25):
    assert scale >= 1.0, "scale should be >= 1.0"
    pad_scale = scale - 1.0
    h, w = np_image.shape[:2]
    top = bottom = int(h * pad_scale)
    left = right = int(w * pad_scale)
    return cv2.copyMakeBorder(np_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128)), (left, top)


def sample_video_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


class FaceSimFID_Evaluator:
    def __init__(
        self, 
        device="cuda",
        face_model_path = "./ckpts",
        inception_weights_path = "./ckpts/face_encoder/inception_v3_google-0cc3c7bd.pth",
    ):
        self.device = device
        if not os.path.exists(face_model_path):
            print("Model not found, downloading from Hugging Face...")
            snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir=face_model_path)
        else:
            print(f"Model already exists in {face_model_path}, skipping download.")
            
        face_arc_path = os.path.join(face_model_path, "face_encoder")
        face_cur_path = os.path.join(face_arc_path, "glint360k_curricular_face_r101_backbone.bin")

        # Initialize FaceEncoder model for face detection and embedding extraction
        face_arc_model = FaceAnalysis(root=face_arc_path, providers=['CUDAExecutionProvider'])
        face_arc_model.prepare(ctx_id=0, det_size=(320, 320))

        # Load face recognition model
        face_cur_model = get_model('IR_101')([112, 112])
        face_cur_model.load_state_dict(torch.load(face_cur_path, map_location="cpu"))
        face_cur_model = face_cur_model.to(device)
        face_cur_model.eval()
        self.face_arc_model = face_arc_model
        self.face_cur_model = face_cur_model
        
        if os.path.exists(inception_weights_path):
            # 手动加载状态字典
            state_dict = torch.load(inception_weights_path, map_location=device)
            fid_model = models.inception_v3(weights=None)  # 不加载预训练权重
            fid_model.load_state_dict(state_dict)
            
            fid_model.fc = torch.nn.Identity()  # Remove final classification layer
            fid_model.eval()
            fid_model = fid_model.to(device)
            
            self.fid_model = fid_model
        else:
            # 如果本地文件也不存在，创建文件并提示手动下载
            print(f"Local weights file not found at {inception_weights_path}")
            print("Please manually download the file from:")
            print("https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth")
            print(f"And save it to: {inception_weights_path}")
            raise ValueError(f"Local weights file not found at {inception_weights_path}")
    
    def get_activations(self, images, batch_size=16):
        activations = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                pred = self.fid_model(batch)
                activations.append(pred)
        activations = torch.cat(activations, dim=0).cpu().numpy()
        if activations.shape[0] == 1:
            activations = np.repeat(activations, 2, axis=0)
        return activations

    def calculate_fid(self, real_activations, fake_activations):
        real_activations_tensor = torch.tensor(real_activations).to(self.device)
        fake_activations_tensor = torch.tensor(fake_activations).to(self.device)

        mu1 = real_activations_tensor.mean(dim=0)
        sigma1 = torch.cov(real_activations_tensor.T)
        mu2 = fake_activations_tensor.mean(dim=0)
        sigma2 = torch.cov(fake_activations_tensor.T)

        ssdiff = torch.sum((mu1 - mu2) ** 2)
        covmean = matrix_sqrt(sigma1.mm(sigma2))
        if torch.is_complex(covmean):
            covmean = covmean.real
        fid = ssdiff + torch.trace(sigma1 + sigma2 - 2 * covmean)
        return fid.item()

    @torch.no_grad()
    def inference(self, img):
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        img.div_(255).sub_(0.5).div_(0.5)
        embedding = self.face_cur_model(img).detach().cpu().numpy()[0]
        return embedding / np.linalg.norm(embedding)

    @torch.no_grad()
    def get_face_keypoints(self, image_bgr):
        face_info = self.face_arc_model.get(image_bgr)
        if len(face_info) > 0:
            return sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
        return None

    def process_image(self, image_path):
        if isinstance(image_path, str) or isinstance(image_path, Path):
            np_faceid_image = np.array(Image.open(image_path).convert("RGB"))
        elif isinstance(image_path, np.ndarray):
            np_faceid_image = image_path
        else:
            raise TypeError("image_path should be a string or PIL.Image.Image object")

        image_bgr = cv2.cvtColor(np_faceid_image, cv2.COLOR_RGB2BGR)

        face_info = self.get_face_keypoints(image_bgr)
        if face_info is None:
            padded_image, sub_coord = pad_np_bgr_image(image_bgr)
            face_info = self.get_face_keypoints(padded_image)
            if face_info is None:
                print("Warning: No face detected in the image. Continuing processing...")
                return None, None
            face_kps = face_info['kps']
            face_kps -= np.array(sub_coord)
        else:
            face_kps = face_info['kps']
        arcface_embedding = face_info['embedding']

        norm_face = face_align.norm_crop(image_bgr, landmark=face_kps, image_size=224)
        align_face = cv2.cvtColor(norm_face, cv2.COLOR_BGR2RGB)

        return align_face, arcface_embedding

    def batch_cosine_similarity(self, embedding_image, embedding_frames):
        embedding_image = torch.tensor(embedding_image).to(self.device)
        embedding_frames = torch.tensor(embedding_frames).to(self.device)
        return torch.nn.functional.cosine_similarity(embedding_image, embedding_frames, dim=-1).cpu().numpy()

    def process_video(
        self, 
        video_path, 
        arcface_image_embedding, 
        cur_image_embedding, 
        real_activations
    ):
        video_frames = sample_video_frames(video_path, num_frames=16)

        # Initialize lists to store the scores
        cur_scores = []
        arc_scores = []
        fid_face = []

        for frame in video_frames:
            # Convert to RGB once at the beginning
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame for ArcFace embeddings
            align_face_frame, arcface_frame_embedding = self.process_image(frame_rgb)

            # Skip if alignment fails
            if align_face_frame is None:
                continue

            # Perform inference for current face model
            cur_embedding_frame = self.inference(align_face_frame)

            # Compute cosine similarity for cur_score and arc_score in a compact manner
            cur_score = max(
                0.0, 
                self.batch_cosine_similarity(cur_image_embedding, cur_embedding_frame).item()
            )
            arc_score = max(
                0.0, 
                self.batch_cosine_similarity(arcface_image_embedding, arcface_frame_embedding).item()
            )

            # Process FID score
            align_face_frame_pil = Image.fromarray(align_face_frame)
            fake_image = load_image(align_face_frame_pil).to(self.device)
            fake_activations = self.get_activations(fake_image)
            fid_score = self.calculate_fid(real_activations, fake_activations)

            # Collect scores
            fid_face.append(fid_score)
            cur_scores.append(cur_score)
            arc_scores.append(arc_score)

        # Aggregate results with default values for empty lists
        avg_cur_score = np.mean(cur_scores) if cur_scores else 0.0
        avg_arc_score = np.mean(arc_scores) if arc_scores else 0.0
        avg_fid_score = np.mean(fid_face) if fid_face else 0.0

        return avg_cur_score, avg_arc_score, avg_fid_score
    
    
    def process(self, image_path, video_path):
        #print(f"image path: {image_path}, type: {type(image_path)}")
        align_face_image, arcface_image_embedding = self.process_image(image_path)
        if align_face_image is None:
            print(f"Error processing image at {image_path}")
            return

        cur_image_embedding = self.inference(align_face_image)
        align_face_image_pil = Image.fromarray(align_face_image)
        real_image = load_image(align_face_image_pil).to(self.device)
        real_activations = self.get_activations(real_image)

        # Process the video and calculate scores
        cur_score, arc_score, fid_score = self.process_video(
            video_path,
            arcface_image_embedding, 
            cur_image_embedding, 
            real_activations,
        )
        return {
            "cur_score": cur_score,
            "arc_score": arc_score,
            "fid_score": fid_score,
        }
    

if __name__ == "__main__":
    video_path = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/new_output_bench/00001_frame/42_0000.mp4"
    image_path = "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/benchmark1/processed/frames/00001_frame.jpg"
    results_file_path = "facesim_fid_score.txt"
    
    facesim_fid_evaluator = FaceSimFID_Evaluator(
        device="cuda",
        face_model_path="./ckpts",
        inception_weights_path="./ckpts/face_encoder/inception_v3_google-0cc3c7bd.pth"
    )
    
    result = facesim_fid_evaluator.process(image_path, video_path)
    cur_score, arc_score, fid_score = result['cur_score'], result['arc_score'], result['fid_score']
    
    # Write results to file
    with open(results_file_path, 'w') as f:
        f.write(f"cur score: {cur_score}\n")
        f.write(f"arc score: {arc_score}\n")
        f.write(f"fid score: {fid_score}\n")

    # Print results
    print(f"cur score: {cur_score}")
    print(f"arc score: {arc_score}")
    print(f"fid score: {fid_score}")
    
    
    
