import os
import argparse
import torch
import pickle
import json
from tqdm import tqdm
import random
import numpy as np
from src.prc import KeyGen, Encode, str_to_bin, bin_to_str
import src.pseudogaussians as prc_gaussians

parser = argparse.ArgumentParser('Args')
parser.add_argument('--hight', type=int, default=64)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--test_num', type=int, default=10)
parser.add_argument('--method', type=str, default='prc')
parser.add_argument('--fpr', type=float, default=0.01)
parser.add_argument('--prc_t', type=int, default=3)
args = parser.parse_args()
print(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
method = args.method
nowm = args.nowm
fpr = args.fpr
prc_t = args.prc_t
hight = args.hight
width = args.width # the width of latent space
n = 4 * hight * width
num_bit = 1024 # message_length

exp_id = f"{hight}_{width}_{num_bit}bit"
if method == 'prc':
    if not os.path.exists(f'keys/{exp_id}.pkl'):  # Generate watermark key for the first time and save it to a file
        (encoding_key_ori, decoding_key_ori) = KeyGen(n, message_length=num_bit, g=None, false_positive_rate=fpr, t=prc_t)  # Sample PRC keys
        with open(f'keys/{exp_id}.pkl', 'wb') as f:  # Save the keys to a file
            pickle.dump((encoding_key_ori, decoding_key_ori), f)
        with open(f'keys/{exp_id}.pkl', 'rb') as f:  # Load the keys from a file
            encoding_key, decoding_key = pickle.load(f)
        assert encoding_key[0].all() == encoding_key_ori[0].all()
    else:  # Or we can just load the keys from a file
        with open(f'keys/{exp_id}.pkl', 'rb') as f:
            encoding_key, decoding_key = pickle.load(f)
        print(f'Loaded PRC keys from file keys/{exp_id}.pkl')