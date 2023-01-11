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
import argparse
import codecs
import yaml
import time
import copy
import os
import paddle


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        for k, v in self.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def setdefault(self, k, default=None):
        if k not in self or self[k] is None:
            self[k] = default
            return default
        else:
            return self[k]


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value


def parse_config(cfg_file):
    """Load a config file into AttrDict"""

    def _update_dic(dic, base_dic):
        '''Update config from dic based base_dic
        '''
        base_dic = base_dic.copy()
        dic = dic.copy()

        if dic.get('_inherited_', True) == False:
            dic.pop('_inherited_')
            return dic

        for key, val in dic.items():
            if isinstance(val, dict) and key in base_dic:
                base_dic[key] = _update_dic(val, base_dic[key])
            else:
                base_dic[key] = val
        dic = base_dic
        return dic

    def _parse_from_yaml(path):
        '''Parse a yaml file and build config'''

        with codecs.open(path, 'r', 'utf-8') as file:
            dic = yaml.load(file, Loader=yaml.FullLoader)

        if '_base_' in dic:
            cfg_dir = os.path.dirname(path)
            base_path = dic.pop('_base_')
            base_path = os.path.join(cfg_dir, base_path)
            base_dic = _parse_from_yaml(base_path)
            dic = _update_dic(dic, base_dic)
        return dic

    yaml_dict = _parse_from_yaml(cfg_file)
    
    yaml_config = AttrDict(yaml_dict)

    create_attr_dict(yaml_config)
    return yaml_config 

def load_inference_model(path_prefix,
                         executor,
                         model_filename=None,
                         params_filename=None):
    # Load onnx model to Inference model.
    if path_prefix.endswith('.onnx'):
        inference_program, feed_target_names, fetch_targets = load_onnx_model(
            path_prefix)
        return [inference_program, feed_target_names, fetch_targets]
    # Load Inference model.
    # TODO: clean code
    if model_filename is not None and model_filename.endswith('.pdmodel'):
        model_name = '.'.join(model_filename.split('.')[:-1])
        assert os.path.exists(
            os.path.join(path_prefix, model_name + '.pdmodel')
        ), 'Please check {}, or fix model_filename parameter.'.format(
            os.path.join(path_prefix, model_name + '.pdmodel'))
        assert os.path.exists(
            os.path.join(path_prefix, model_name + '.pdiparams')
        ), 'Please check {}, or fix params_filename parameter.'.format(
            os.path.join(path_prefix, model_name + '.pdiparams'))
        model_path_prefix = os.path.join(path_prefix, model_name)
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(
                path_prefix=model_path_prefix, executor=executor))
    elif model_filename is not None and params_filename is not None:
        [inference_program, feed_target_names, fetch_targets] = (
            paddle.static.load_inference_model(
                path_prefix=path_prefix,
                executor=executor,
                model_filename=model_filename,
                params_filename=params_filename))
    else:
        model_name = '.'.join(model_filename.split('.')
                              [:-1]) if model_filename is not None else 'model'
        if os.path.exists(os.path.join(path_prefix, model_name + '.pdmodel')):
            model_path_prefix = os.path.join(path_prefix, model_name)
            [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(
                    path_prefix=model_path_prefix, executor=executor))
        else:
            [inference_program, feed_target_names, fetch_targets] = (
                paddle.static.load_inference_model(
                    path_prefix=path_prefix, executor=executor))

    #import pdb; pdb.set_trace()
    return [inference_program, feed_target_names, fetch_targets]
