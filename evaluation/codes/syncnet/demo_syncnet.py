#!/usr/bin/python
#-*- coding: utf-8 -*-

import time, pdb, argparse, subprocess

from SyncNetInstance import *

# ==================== LOAD PARAMS ====================


parser = argparse.ArgumentParser(description = "SyncNet");

parser.add_argument('--initial_model', type=str, default="/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/mm1/Lab/custom/ConsisID/eval/new_metric/Wav2Lip/evaluation/syncnet_python/data/syncnet_v2.model", help='');
parser.add_argument('--batch_size', type=int, default='20', help='');
parser.add_argument('--vshift', type=int, default='15', help='');
parser.add_argument('--videofile', type=str, default="data/example.avi", help='');
parser.add_argument('--tmp_dir', type=str, default="data/work/pytmp", help='');
parser.add_argument('--reference', type=str, default="demo", help='');

opt = parser.parse_args();


# ==================== RUN EVALUATION ====================

s = SyncNetInstance();

s.loadParameters(opt.initial_model);
print("Model %s loaded."%opt.initial_model);

s.evaluate(opt, videofile=opt.videofile)