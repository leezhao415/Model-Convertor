#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
from loguru import logger

import torch
from torch import nn
import cv2
import numpy as np

from ModelFile.yolox.build import get_exp
from ModelFile.yolox.network_blocks import SiLU
from ModelFile.yolox.model_utils import replace_module

import onnxruntime
from ModelFile.yolox.data_augment import preproc as preprocess
from ModelFile.yolox.voc_classes import VOC_CLASSES
from ModelFile.yolox.demo_utils import mkdir, multiclass_nms, demo_postprocess
from ModelFile.yolox.visualize import vis

def make_parser():
    parser = argparse.ArgumentParser("YOLOX onnx deploy")
    ## ----------------ONNX model-------------------------------------------------------------------------------------------------- ##
    parser.add_argument(
        "--output-name", type=str, default="./Outputs/model_onnx/model.onnx", help="output name of models")
    parser.add_argument(
        "--output-name-sim", type=str, default="./Outputs/model_onnx_sim/model_sim.onnx", help="output name of models-sim")
    # config_file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='./ModelFile/yolox/yolox_voc_nano.py',
        type=str,
        help="expriment description file",)
    # model_file
    parser.add_argument("-c", "--ckpt", default='./ModelFile/model.pth', type=str, help="ckpt path")
    ## -------------------------------------------------------------------------------------------------------------------------------- ##
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="Outputs", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument(
        "--dynamic", action="store_true", help="whether the input shape should be dynamic or not"
    )
    parser.add_argument("--no-onnxsim", action="store_true", help="use onnxsim or not")
    
    ## ----------------ONNX model_infer------------------------------------------------------------------------------------------ ##
    # Path to your input image.
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        default='./TestFiles/obj365_train_000000026463.jpg',
        help="Path to your input image.",)
    # Path to your output directory.
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./Outputs/demo_output_onnx_infer',
        help="Path to your output directory.",)
    # Score threshould
    parser.add_argument(
        "-s",
        "--score_thr",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",)
    # input shape
    parser.add_argument(
        "--input_shape",
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",)
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",)
    ## -------------------------------------------------------------------------------------------------------------------------------- ##
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()
    print('*' * 60)
    print("参数详情: {}".format(args))
    print('*' * 60)
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    model = exp.get_model()
    if args.ckpt is None:
        file_name = os.path.join(exp.output_dir, args.experiment_name)
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt

    # load the model state dict
    ckpt = torch.load(ckpt_file, map_location="cpu")

    model.eval()
    if "model" in ckpt:
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    logger.info("loading checkpoint done.")
    dummy_input = torch.randn(args.batch_size, 3, exp.test_size[0], exp.test_size[1])

    torch.onnx._export(
        model,
        dummy_input,
        args.output_name,
        input_names=[args.input],
        output_names=[args.output],
        dynamic_axes={args.input: {0: 'batch'},
                      args.output: {0: 'batch'}} if args.dynamic else None,
        opset_version=args.opset,
    )
    logger.info("主人，ONNX模型已生成，名称为 {}，请查阅！".format(args.output_name))
 
    # ONNX-sim model
    if not args.no_onnxsim:
        import onnx

        from onnxsim import simplify

        input_shapes = {args.input: list(dummy_input.shape)} if args.dynamic else None

        # use onnxsimplify to reduce reduent model.
        onnx_model = onnx.load(args.output_name)
        model_simp, check = simplify(onnx_model,
                                     dynamic_input_shape=args.dynamic,
                                     input_shapes=input_shapes)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, args.output_name_sim)
        logger.info("主人，ONNX模型简化已完成，名称为 {}，请查阅！".format(args.output_name_sim))
    
    # ONNX model_infer
    input_shape = tuple(map(int, args.input_shape.split(',')))
    origin_img = cv2.imread(args.image_path)
    img, ratio = preprocess(origin_img, input_shape)

    session = onnxruntime.InferenceSession(args.output_name_sim)

    ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
    output = session.run(None, ort_inputs)
    predictions = demo_postprocess(output[0], input_shape, p6=args.with_p6)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                         conf=args.score_thr, class_names=VOC_CLASSES)

    mkdir(args.output_dir)
    output_path = os.path.join(args.output_dir, args.image_path.split("/")[-1])
    cv2.imwrite(output_path, origin_img)
    logger.info('主人，ONNX模型推理已完成，请查阅！')

if __name__ == "__main__":
    main()
