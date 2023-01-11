import os
import sys
import random
import numpy as np
import argparse
import time

import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.quant.analysis_qat import AnalysisQAT
from utils import parse_config, load_inference_model
from paddle.fluid.contrib.slim.quantization import PostTrainingQuantization
#print(paddle.__file__)
from data import create_eval_dataset
from paddlenlp.transformers.gpt.tokenizer import GPTTokenizer
from analysis_ptq import AnalysisPTQ

def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='analysis_results',
        help="directory to save compressed model.")
    parser.add_argument(
        '--pretrain_dir',
        type=str,
        default='pretrained model dir',
        help="directory to pretrained model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    return parser

def reader_wrapper(reader, input_list):
    def gen():
        for data in reader:
            tokens, loss_mask, attention_mask, position_ids, labels = data
            in_dict = {}
            in_dict['tokens'] = tokens
            in_dict['ids'] = position_ids
            yield in_dict

    return gen

def eval_reader_wrapper(reader):
    def gen():
        for data in reader:
            tokens, loss_mask, attention_mask, position_ids, labels = data
            in_dict = {}
            in_dict['tokens'] = tokens
            in_dict['ids'] = position_ids
            yield in_dict, labels, loss_mask

    return gen


def eval_function(exe, program, feed_names, fetch_list):
    tic_eval = time.time()
    score_name = "loss" if not global_config['cloze_eval'] else "number correct"
    first_step = True
    eval_losses = []
    total_score = 0
    for eval_step, (data, labels, loss_mask) in enumerate(eval_loader()):
        #print(len(data))
        preds = exe.run(program=program,
                        feed=data,
                        fetch_list=fetch_list,
                        return_numpy=False)

        paddle.disable_static()

        labels = paddle.to_tensor(labels)
        preds = paddle.to_tensor(preds[0])
        loss_mask = paddle.to_tensor(loss_mask)

        if not global_config['cloze_eval']:
            if first_step:
                first_step = False

            masked_lm_loss = paddle.nn.functional.cross_entropy(
                preds, labels, reduction="none")
            loss = paddle.sum(masked_lm_loss * loss_mask)
            eval_losses.append(loss.numpy()[0])
            total_score += loss.numpy() / (num_tokenized_tokens - 1)

        else:
            if first_step:
                first_step = False
            outputs = paddle.argmax(preds, -1)
            acc = paddle.cast(outputs == labels, 'float32')
            acc = paddle.where(
                paddle.cast(loss_mask, 'bool'), acc, paddle.ones_like(acc))
            acc = paddle.sum(paddle.prod(acc, -1))
            eval_losses.append(acc.numpy()[0])
            total_score += acc.numpy()[0]

        if eval_step != 0 and (eval_step % 10 == 0):
            print("[eval] step: %d, batch: %d, %s: %.9f, speed: %.2f step/s" %
                  (eval_step, eval_step, score_name, total_score,
                   1. / (time.time() - tic_eval)))
            tic_eval = time.time()
        paddle.enable_static()

    metric = None
    if not global_config['cloze_eval']:
        total_loss = float(total_score)
        ppl = math.exp(min(20, total_loss))
        token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
        adjusted_ppl = math.exp(min(20, total_loss * token_ratio))
        string = ' validation results on {} | '.format(gpt_config['Data'][
            'Eval']['dataset']['name'])
        string += 'avg loss: {:.4E} | '.format(total_loss)
        string += 'ppl: {:.4E} | '.format(ppl)
        string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
        string += 'token ratio: {} |'.format(token_ratio)
        metric = ppl
    else:
        num_correct = float(total_score)
        acc = float(num_correct / num_examples)
        string = ' validation results on {} | '.format(gpt_config['Data'][
            'Eval']['dataset']['name'])
        string += 'number correct: {:.4E} | '.format(num_correct)
        string += 'total examples: {:.4E} | '.format(num_examples)
        string += 'avg accuracy: {:.4E}'.format(acc)
        metric = acc

    print(string)
    return metric

def main():
    global global_config, all_config
    all_config = load_slim_config(FLAGS.config_path)
    assert "Global" in all_config, "Key 'Global' not found in config file. \n{}".format(
        all_config)
    global_config = all_config["Global"]
    ptq_config = all_config['PTQ']
    seed = all_config['Global']['seed']
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)

    global gpt_config
    gpt_config = parse_config(global_config['reader_config'])

    if not global_config['cloze_eval']:
        gpt_config['Data']['Eval']['dataset']['name'] = "LM_Eval_Dataset"
    else:
        gpt_config['Data']['Eval']['dataset']['name'] = "Lambada_Eval_Dataset"

    tokenizer = GPTTokenizer.from_pretrained(FLAGS.pretrain_dir)
    global num_examples
    num_examples = 0.0
    valid_data_loader, num_original_tokens,num_tokenized_tokens,num_examples = create_eval_dataset(gpt_config['Data'], tokenizer)

    global eval_loader
    eval_loader = eval_reader_wrapper(valid_data_loader)

    data_loader = reader_wrapper(valid_data_loader, ['input_ids'])

    place = paddle.CUDAPlace(0) if global_config['device'] == 'gpu' else paddle.CPUPlace()
    exe = paddle.static.Executor(place)

    skip_tensor_list = []
    for i in range(500):
        skip_tensor_list.append('transpose_{}.tmp_0'.format(i))
    skip_tensor_list.append('linear_291.tmp_0')
    skip_tensor_list.append('tokens')
    skip_tensor_list.append('embedding_1.tmp_0')
    skip_tensor_list.append('embedding_0.w_0')
    skip_tensor_list.append('tmp_0')
    skip_tensor_list.append('layer_norm_97.tmp_2')
    #ptq_config['skip_tensor_list'] = skip_tensor_list
    
    analysis_ptq = AnalysisPTQ(model_dir=FLAGS.pretrain_dir,
            model_filename='model.pdmodel',
            params_filename='model.pdiparams',
            eval_function=None,
            data_loader=data_loader,
            save_dir='analysis_results',
            ptq_config=ptq_config
            #resume=True,
            )
    analysis_ptq.statistical_analyse()

if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()
    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
