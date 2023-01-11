# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import math
import time
import sys
import random
import numpy as np
import yaml

import paddle
from paddle.io import DataLoader
import paddleslim
from log import logger

#sys.path.append("../../../")
from ppfleetx.models.language_model.gpt import GPTModel, GPTForPretraining
from ppfleetx.data.tokenizers import GPTTokenizer
from ppfleetx.data.sampler import Stack, Tuple
from ppfleetx.data.dataset import LM_Eval_Dataset, Lambada_Eval_Dataset
from paddleslim.quant import quant_post_static
from paddleslim.common import load_inference_model
from paddleslim.quant.analysis_ptq import AnalysisPTQ
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantizationProgram
from paddle.fluid.executor import global_scope

from tools import parse_yaml, wikitext_detokenizer 

def get_tokens(tokenizer, text, strict=True):
    if not strict:
        tokens = tokenizer.encode(text)
        return tokens[:-1], [tokens[-1]]
    last_token = text.split()[-1]
    start_idx = text.rfind(last_token)
    beginning_tokens = tokenizer.encode(text[:start_idx].strip())
    last_token = tokenizer.encode(' ' + last_token)
    return beginning_tokens, last_token


def create_eval_dataset(configs):
    val_dataloader = None
    eval_batch_size = configs['batch_size']
    seq_len = configs['max_seq_len']

    tokenizer = GPTTokenizer.from_pretrained("gpt2")
    if not configs['cloze_eval']:
        with open(configs['eval_path'], "rb") as reader:
            entire_data = reader.read().decode('utf-8')
        num_original_tokens = len(entire_data.strip().split(" "))
        entire_data = wikitext_detokenizer(entire_data)
        tokenized_data = tokenizer.encode(entire_data)
        num_tokenized_tokens = len(tokenized_data)
        print('Original Tokens: %d, Detokenized tokens: %d' %
              (num_tokenized_tokens, num_original_tokens))
        val_dataset = LM_Eval_Dataset(tokenized_data, seq_len,
                                      tokenizer.eos_token_id,
                                      configs['overlapping_eval'])
    else:
        tokenized_data = []
        tokenized_label = []
        with open(configs['eval_path'], 'r') as f:
            for line in f.readlines():
                text = json.loads(line)['text']
                tokens, labels = get_tokens(tokenizer, text)
                tokenized_data.append(tokens)
                tokenized_label.append(labels)
        val_dataset = Lambada_Eval_Dataset(
            tokenized_data, tokenized_label, seq_len,
            tokenizer.eos_token_id)  #tokenizer.pad_token_id)
        num_tokenized_tokens = 0
        num_original_tokens = 0

    val_dict = {
        'num_examples': len(val_dataset),
        'num_original_tokens': num_original_tokens,
        'num_tokenized_tokens': num_tokenized_tokens,
    }

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        drop_last=False,
        collate_fn=Tuple(Stack(), Stack(), Stack(), Stack(), Stack()))

    return val_dataloader, val_dict


def eval_reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            tokens, loss_mask, attention_mask, position_ids, labels = data
            in_dict = {}
            in_dict['tokens'] = tokens
            in_dict['ids'] = position_ids
            yield in_dict, labels, loss_mask
    return gen

def reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            tokens, loss_mask, attention_mask, position_ids, labels = data
            in_dict = {}
            in_dict['tokens'] = tokens
            in_dict['ids'] = position_ids
            yield in_dict

    return gen

_cnt = 0

def counter():
    global _cnt
    _cnt += 1
    return _cnt


def eval_function(exe, program, feed_names, fetch_list):
    total_score = 0
    tic_eval = time.time()
    score_name = "loss" if not configs['Eval']['cloze_eval'] else "number correct"
    step = 0
    for data, labels, loss_mask in eval_loader():
        step += 1 
        preds = exe.run(program=program,
                       feed=data,
                       fetch_list=fetch_list,
                       return_numpy=False)
        
        paddle.disable_static()
        
        labels = paddle.to_tensor(labels)
        preds = paddle.to_tensor(preds[0])
        
        loss_mask = paddle.to_tensor(loss_mask)
        if not configs['Eval']['cloze_eval']:
            masked_lm_loss = paddle.nn.functional.cross_entropy(
                preds, labels, reduction="none")
            loss = paddle.sum(masked_lm_loss * loss_mask)
            total_score += loss.numpy() / (
                eval_dict['num_tokenized_tokens'] - 1)

        else:
            outputs = paddle.argmax(preds, -1)
            acc = paddle.cast(outputs == labels, 'float32')
            acc = paddle.where(
                paddle.cast(loss_mask, 'bool'), acc, paddle.ones_like(acc))
            acc = paddle.sum(paddle.prod(acc, -1))
            total_score += acc.numpy()

        if step % configs['Eval']['logging_freq'] == 0:
            logger.info(
                "[eval] step %d, batch: %d, %s: %f, speed: %.2f step/s" %
                (step, step, score_name, total_score,
                 configs['Eval']['logging_freq'] /
                 (time.time() - tic_eval)))
            tic_eval = time.time()
        paddle.enable_static()

    metric = None
    if not configs['Eval']['cloze_eval']:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (eval_dict['num_tokenized_tokens'] - 1) / (
            eval_dict['num_original_tokens'] - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = ' validation results on {} | '.format(configs['Eval'][
            'eval_path'])
        string += 'avg loss: {:.4E} | '.format(total_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
        metric = ppl
    else:
        num_correct = float(total_score)
        acc = float(num_correct / eval_dict['num_examples'])
        string = ' validation results on {} | '.format(configs['Eval'][
            'eval_path'])
        string += 'number correct: {:.4E} | '.format(num_correct)
        string += 'total examples: {:.4E} | '.format(eval_dict['num_examples'])
        string += 'avg accuracy: {:.4E}'.format(acc)
        metric = acc
    
    logger.info(string)
    return acc
    

ptq_config = {
  'quantizable_op_type': [ "mul", "matmul", "matmul_v2"],
  'weight_quantize_type': 'abs_max',
  'activation_quantize_type': 'moving_average_abs_max',
  'is_full_quantize': False,
  'batch_size': 1,
  'batch_nums': 1
}

def do_ptq():
    global configs
    configs = parse_yaml()
    paddle.set_device(configs['Global']['device'])

    seed = configs['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    global eval_dict
    eval_data_loader, eval_dict = create_eval_dataset(configs['Eval'])
    paddle.enable_static()

    data_loader = reader_wrapper(eval_data_loader, ['input_ids'])
    global eval_loader
    eval_loader = eval_reader_wrapper(eval_data_loader, ['input_ids'])
    
    place = paddle.CUDAPlace(0) if configs['Global']['device'] == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)
    
#     print('Loading inference model')
#     [program, feed_list, fetch_list]= load_inference_model( \
#             './base_gpt_345M', \
#             executor=exe, \
#             model_filename='model.pdmodel', \
#             params_filename='model.pdiparams')
    
#     print('Loaded inference model')
    
#     from paddle.fluid import core
#     from paddle.fluid.framework import IrGraph
#     graph = IrGraph(core.Graph(program.desc), for_test=True)
#     ops = graph.all_op_nodes()
#     input_rename_map = {}
#     output_rename_map = {}
#     for op_node in ops:
#         op_name = op_node.name()
#         if op_name == 'dropout':
#             out_var = graph._find_node_by_name(op_node.outputs, op_node.output('Out')[0])
#             in_var = graph._find_node_by_name(op_node.inputs, op_node.input('X')[0])
#             input_rename_map[out_var.node] = in_var
#             output_rename_map[in_var.node] = out_var
#             graph.safe_remove_nodes(op_node)
#     ops = graph.all_op_nodes()
#     for op_node in ops:
#         for var in op_node.inputs:
#             if var.node in input_rename_map:
#                 old_in = var
#                 new_in = input_rename_map[var.node]
#                 graph.update_input_link(old_in, new_in, op_node)
#                 print(f'relink {op_node.name()} \'s input node from {old_in.name()} to {new_in.name()}.')
#         for var in op_node.outputs:
#             if var.node in output_rename_map:
#                 old_out = var
#                 new_out = output_rename_map[var.node]
#                 graph.update_input_link(old_out, new_out, op_node)
#                 print(f'relink {op_node.name()} \'s output node from {old_out.name()} to {new_out.name()}.')
    
#     program = graph.to_program()
#     paddle.fluid.io.save_inference_model(
#                     dirname='base_gpt_345M_nodrop',
#                     model_filename='model.pdmodel',
#                     params_filename='model.pdiparams',
#                     feeded_var_names=feed_list,
#                     target_vars=fetch_list,
#                     executor=exe,
#                     main_program=program
#                 )
    
    from paddleslim.quant import quant_post_static
    from paddleslim.quant import quant_recon_static
    quant_recon_static(
        executor=exe,
        model_dir='./base_gpt_345M/',
        quantize_model_path='./rank_0/recon',
        data_loader=data_loader,
        model_filename='model.pdmodel',
        params_filename='model.pdiparams',
        batch_size=1,
        batch_nums=10,
        algo='avg',
        hist_percent=0.999,
        is_full_quantize=False,
        bias_correction=False,
        onnx_format=False,
        weight_quantize_type='channel_wise_abs_max',
        recon_level='layer-wise',
        simulate_activation_quant=False,
        # regions=config['regions'],
        # region_weights_names=config['region_weights_names'],
        skip_tensor_list=['embedding_0.w_0', 'linear_15.w_0', 'linear_11.w_0', 'linear_31.w_0',
                          'linear_27.w_0', 'linear_39.w_0', 'linear_83.w_0', 'linear_87.w_0', 'linear_91.w_0',
                          'linear_19.w_0', 'linear_43.w_0', 'linear_22.w_0'],
        # if 'skip_tensor_list' in config else None,
        epochs=10,
        lr=0.1)
    
#     analyzer = AnalysisQuant(
#         model_dir='./rank_2',
#         model_filename='model.pdmodel',
#         params_filename='model.pdiparams',
#         eval_function=eval_function,
#         data_loader=data_loader,
#         save_dir='analysis_gpt',
#         ptq_config=ptq_config,
#         resume=True, )

#     analyzer.statistical_analyse()
#     analyzer.metric_error_analyse()
    
#     post_training_quantization = PostTrainingQuantizationProgram(
#             exe,
#             program,
#             freeze_model=False,
#             return_graph=True,
#             batch_size=2,
#             batch_nums=128,
#             algo="avg",
#             activation_quantize_type='moving_average_abs_max',
#             weight_quantize_type='abs_max',
#             data_loader=data_loader,
#             feed_list=feed_list,
#             fetch_list=fetch_list,
#             quantizable_op_type=[
#                           "conv2d", "depthwise_conv2d", "mul", "matmul",
#                           "matmul_v2"
#                       ],
#             skip_tensor_list=['embedding_0.w_0', 'linear_15.w_0', 'linear_11.w_0', 'linear_31.w_0',
#                               'linear_27.w_0', 'linear_39.w_0', 'linear_83.w_0', 'linear_87.w_0', 'linear_91.w_0',
#                          'linear_19.w_0', 'linear_43.w_0'],
#             onnx_format=True,
#     )
#     main_graph = post_training_quantization.quantize()

        
#     post_training_quantization.save_quantized_model(
#         './rank_0/quantized_skip_1',
#         model_filename='model.pdmodel',
#         params_filename='model.pdiparams')
    
    # eval_function(exe, program, feed_list, fetch_list)
    
    






if __name__ == "__main__":
    do_ptq()
