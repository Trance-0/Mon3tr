import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'mast3r'))
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), 'mast3r'),'dust3r'))

import torch
from mast3r.model import AsymmetricMASt3R
from mast3r.demo import get_args_parser as mast3r_get_args_parser
from mon3tr.demo import main_demo
from PIL import Image
import numpy as np
import gradio as gr
import tempfile
from contextlib import nullcontext

from mast3r.utils.misc import hash_md5

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.demo import set_print_with_timestamp

import matplotlib.pyplot as pl
pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

def get_args_parser():
    parser = mast3r_get_args_parser()
    # change defaults
    parser.prog = 'Mon3tr demo'
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # change default tmp_dir to the current directory
    if args.tmp_dir is None:
        args.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')

    set_print_with_timestamp()

    if args.server_name is not None:
        server_name = args.server_name
    else:
        server_name = '0.0.0.0' if args.local_network else '127.0.0.1'

    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)
    chkpt_tag = hash_md5(weights_path)

    def get_context(tmp_dir):
        return tempfile.TemporaryDirectory(suffix='_mon3tr_gradio_demo') if tmp_dir is None \
            else nullcontext(tmp_dir)
    with get_context(args.tmp_dir) as tmpdirname:
        cache_path = os.path.join(tmpdirname, chkpt_tag)
        os.makedirs(cache_path, exist_ok=True)
        main_demo(cache_path, model, args.retrieval_model, args.device, args.image_size, server_name, args.server_port,
                  silent=args.silent, share=args.share, gradio_delete_cache=args.gradio_delete_cache)
