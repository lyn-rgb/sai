#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy as np
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree


# ==================== Get OFFSET ====================

def calc_pdist(feat1, feat2, vshift=10):
    
    win_size = vshift*2+1

    feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

    dists = []

    for i in range(0,len(feat1)):

        dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

    return dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
        super(SyncNetInstance, self).__init__();

        self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

    def evaluate(self, opt, videofile):

        self.__S__.eval();

        # ========== ==========
        # Convert files
        # ========== ==========

        if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
          rmtree(os.path.join(opt.tmp_dir,opt.reference))

        os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

        command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'%06d.jpg'))) 
        output = subprocess.call(command, shell=True, stdout=None)

        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))) 
        output = subprocess.call(command, shell=True, stdout=None)
        
        # ========== ==========
        # Load video 
        # ========== ==========

        images = []
        
        flist = glob.glob(os.path.join(opt.tmp_dir,opt.reference,'*.jpg'))
        flist.sort()

        # SyncNet v2 is trained on 224x224 BGR face crops with raw 0-255 pixel scale.
        target_size = (224, 224)
        
        print(f"Loading {len(flist)} images...")
        
        for fname in flist:
            img = cv2.imread(fname)
            if img is None:
                print(f"Warning: Could not read image {fname}")
                continue
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            images.append(img)

        if len(images) == 0:
            print("Error: No images loaded!")
            return None, None, None

        print(f"Loaded {len(images)} images")
        
        # Match the original SyncNet demo tensor layout: [B, C, T, H, W].
        im = np.stack(images, axis=3)
        print(f"After stack: {im.shape}")

        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))
        print(f"After transpose: {im.shape}")
        
        imtv = torch.autograd.Variable(torch.from_numpy(im).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
        mfcc_features = python_speech_features.mfcc(audio, sample_rate)
        mfcc_features = np.stack([np.array(item) for item in zip(*mfcc_features)])
        
        print(f"MFCC shape: {mfcc_features.shape}")
        
        # 添加batch和channel维度 -> (1, 1, 特征维度, 时间步)
        cc = np.expand_dims(np.expand_dims(mfcc_features, axis=0), axis=0)
        print(f"Audio input shape: {cc.shape}")
        
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(np.float32)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        video_length = float(len(images))/25
        audio_length = float(len(audio))/16000
        mfcc_length = mfcc_features.shape[1] * 0.01  # 每个MFCC帧对应10ms
        
        print(f"Video length: {video_length:.4f}s ({len(images)} frames)")
        print(f"Audio length: {audio_length:.4f}s ({len(audio)} samples)")
        print(f"MFCC length: {mfcc_length:.4f}s ({mfcc_features.shape[1]} frames)")
        
        if abs(audio_length - video_length) > 0.5:
            print(f"WARNING: Audio ({audio_length:.4f}s) and video ({video_length:.4f}s) lengths are significantly different.")

        # 25 fps video and 16 kHz audio: one video frame corresponds to 640 audio samples.
        min_length = min(len(images), math.floor(len(audio) / 640))
        
        if min_length <= 5:
            print("Error: Video too short!")
            return None, None, None
            
        print(f"Min length for processing: {min_length} frames")
        
        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):
            batch_end = min(lastframe, i + opt.batch_size)
            
            # 处理视频批次 - 提取5帧的滑动窗口
            im_batch = [imtv[:, :, vframe:vframe+5, :, :] for vframe in range(i, batch_end)]
            im_in = torch.cat(im_batch, 0)
            print(f"Video batch shape: {im_in.shape}")
            
            im_out = self.__S__.forward_lip(im_in.cuda())
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct[:, :, :, vframe*4:vframe*4+20] for vframe in range(i, batch_end)]
            cc_in = torch.cat(cc_batch, 0)
            print(f"Audio batch shape: {cc_in.shape}")
            
            cc_out = self.__S__.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)
        
        print(f"Final video features shape: {im_feat.shape}")
        print(f"Final audio features shape: {cc_feat.shape}")

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        dists = calc_pdist(im_feat, cc_feat, vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1)

        minval, minidx = torch.min(mdist, 0)

        offset = opt.vshift - minidx
        conf = torch.median(mdist) - minval

        offset_val = int(offset)
        conf_val = float(conf)
        min_val = float(minval)

        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset_val, min_val, conf_val))

        return torch.tensor([offset_val]), torch.tensor([conf_val]), torch.tensor([min_val])

    def extract_feature(self, opt, videofile):

        self.__S__.eval();
        
        # ========== ==========
        # Load video 
        # ========== ==========
        cap = cv2.VideoCapture(videofile)

        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.__S__.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        return im_feat


    def loadParameters(self, path):
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

        self_state = self.__S__.state_dict();

        for name, param in loaded_state.items():

            self_state[name].copy_(param);


# #!/usr/bin/python
# #-*- coding: utf-8 -*-
# # Video 25 FPS, Audio 16000HZ

# import torch
# import numpy
# import time, pdb, argparse, subprocess, os, math, glob
# import cv2
# import python_speech_features

# from scipy import signal
# from scipy.io import wavfile
# from SyncNetModel import *
# from shutil import rmtree


# # ==================== Get OFFSET ====================

# def calc_pdist(feat1, feat2, vshift=10):
    
#     win_size = vshift*2+1

#     feat2p = torch.nn.functional.pad(feat2,(0,0,vshift,vshift))

#     dists = []

#     for i in range(0,len(feat1)):

#         dists.append(torch.nn.functional.pairwise_distance(feat1[[i],:].repeat(win_size, 1), feat2p[i:i+win_size,:]))

#     return dists

# # ==================== MAIN DEF ====================

# class SyncNetInstance(torch.nn.Module):

#     def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):
#         super(SyncNetInstance, self).__init__();

#         self.__S__ = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

#     def evaluate(self, opt, videofile):

#         self.__S__.eval();

#         # ========== ==========
#         # Convert files
#         # ========== ==========

#         if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
#           rmtree(os.path.join(opt.tmp_dir,opt.reference))

#         os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

#         command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'%06d.jpg'))) 
#         output = subprocess.call(command, shell=True, stdout=None)

#         command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (videofile,os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))) 
#         output = subprocess.call(command, shell=True, stdout=None)
        
#         # ========== ==========
#         # Load video 
#         # ========== ==========

#         images = []
        
#         flist = glob.glob(os.path.join(opt.tmp_dir,opt.reference,'*.jpg'))
#         flist.sort()

#         for fname in flist:
#             images.append(cv2.imread(fname))

#         im = numpy.stack(images,axis=3)
#         im = numpy.expand_dims(im,axis=0)
#         im = numpy.transpose(im,(0,3,4,1,2))

#         imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

#         # ========== ==========
#         # Load audio
#         # ========== ==========

#         sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
#         mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
#         mfcc = numpy.stack([numpy.array(i) for i in mfcc])

#         cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
#         cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

#         # ========== ==========
#         # Check audio and video input length
#         # ========== ==========

#         if (float(len(audio))/16000) != (float(len(images))/25) :
#             print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(len(audio))/16000,float(len(images))/25))

#         min_length = min(len(images),math.floor(len(audio)/640))
        
#         # ========== ==========
#         # Generate video and audio feats
#         # ========== ==========

#         lastframe = min_length-5
#         im_feat = []
#         cc_feat = []

#         tS = time.time()
#         for i in range(0,lastframe,opt.batch_size):
            
#             im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
#             im_in = torch.cat(im_batch,0)
#             im_out  = self.__S__.forward_lip(im_in.cuda());
#             im_feat.append(im_out.data.cpu())

#             cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
#             cc_in = torch.cat(cc_batch,0)
#             cc_out  = self.__S__.forward_aud(cc_in.cuda())
#             cc_feat.append(cc_out.data.cpu())

#         im_feat = torch.cat(im_feat,0)
#         cc_feat = torch.cat(cc_feat,0)

#         # ========== ==========
#         # Compute offset
#         # ========== ==========
            
#         print('Compute time %.3f sec.' % (time.time()-tS))

#         dists = calc_pdist(im_feat,cc_feat,vshift=opt.vshift)
#         mdist = torch.mean(torch.stack(dists,1),1)

#         minval, minidx = torch.min(mdist,0)

#         offset = opt.vshift-minidx
#         conf   = torch.median(mdist) - minval

#         fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
#         # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
#         fconf   = torch.median(mdist).numpy() - fdist
#         fconfm  = signal.medfilt(fconf,kernel_size=9)
        
#         numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
#         print('Framewise conf: ')
#         print(fconfm)
#         print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

#         dists_npy = numpy.array([ dist.numpy() for dist in dists ])
#         return offset.numpy(), conf.numpy(), dists_npy

#     def extract_feature(self, opt, videofile):

#         self.__S__.eval();
        
#         # ========== ==========
#         # Load video 
#         # ========== ==========
#         cap = cv2.VideoCapture(videofile)

#         frame_num = 1;
#         images = []
#         while frame_num:
#             frame_num += 1
#             ret, image = cap.read()
#             if ret == 0:
#                 break

#             images.append(image)

#         im = numpy.stack(images,axis=3)
#         im = numpy.expand_dims(im,axis=0)
#         im = numpy.transpose(im,(0,3,4,1,2))

#         imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
#         # ========== ==========
#         # Generate video feats
#         # ========== ==========

#         lastframe = len(images)-4
#         im_feat = []

#         tS = time.time()
#         for i in range(0,lastframe,opt.batch_size):
            
#             im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
#             im_in = torch.cat(im_batch,0)
#             im_out  = self.__S__.forward_lipfeat(im_in.cuda());
#             im_feat.append(im_out.data.cpu())

#         im_feat = torch.cat(im_feat,0)

#         # ========== ==========
#         # Compute offset
#         # ========== ==========
            
#         print('Compute time %.3f sec.' % (time.time()-tS))

#         return im_feat


#     def loadParameters(self, path):
#         loaded_state = torch.load(path, map_location=lambda storage, loc: storage);

#         self_state = self.__S__.state_dict();

#         for name, param in loaded_state.items():

#             self_state[name].copy_(param);
