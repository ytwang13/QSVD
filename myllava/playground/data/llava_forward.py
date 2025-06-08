
import os
os.environ['HF_HOME'] = '/vast/yw6594/log'
os.chdir("/scratch/yw6594/cf/vlm/LLaVA")
import argparse
import torch
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from typing import Callable, List, Optional, Sequence, Union, Tuple
from PIL import Image
import math
    # "torch==2.3.1", "torchvision==0.18.1",
model_path = 'liuhaotian/llava-v1.5-13b'
disable_torch_init()
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
# if cpu use bfloat16
if torch.cuda.is_available():
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, torch_dtype=torch.float16)
else:
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, torch_dtype=torch.bfloat16)

# print(str(model))

def count_params_in_gb(model):
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    # Each parameter takes 2 bytes (FP16 precision), convert to GB
    total_params_in_gb = (total_params * 2) / (1024 ** 3)  # 1 GB = 1024^3 bytes
    return total_params_in_gb

print(f'LLava LM params {count_params_in_gb(model.model.layers)}GB')
print(f'LLava VIT params {count_params_in_gb(model.model.vision_tower.vision_tower.vision_model)}GB')
print('finish model params')
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
question_file = './playground/data/eval/scienceqa/llava_test_CQM-A.json'
questions = json.load(open(os.path.expanduser(question_file), "r"))
num_chunks = 1
chunk_idx = 0
questions = get_chunk(questions, num_chunks, chunk_idx)


#img: 5 11 23 42 46 61 62 78  85 86 87
#q:   1 2  5  9  10 13 14 15  16 17 18
#b:   0 1  2  3  4  5  6   7   8  9 10
qidx = 5
line = questions[qidx]
idx = line['id']
question = line['conversations'][0]
qs = question['value'].replace('<image>', '').strip()
cur_prompt = qs
image_folder = './playground/data/eval/scienceqa/images/test'
if 'image' in line:
    image_file = line["image"]
    image = Image.open(os.path.join(image_folder, image_file)) # (760, 506)
    image_tensor = process_images([image], image_processor, model.config)[0]
    try:
        images = image_tensor.unsqueeze(0).half().cuda()
    except:
        images = image_tensor.unsqueeze(0).to(torch.bfloat16)
        print("Now image on cpu!")
    # images = image_tensor.unsqueeze(0).to(torch.bfloat16)
    image_sizes = [image.size]
    if getattr(model.config, 'mm_use_im_start_end', False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs # use here
    cur_prompt = '<image>' + '\n' + cur_prompt

qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."



conv = conv_templates['vicuna_v1'].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
##### tokenize sentence and put image_index into <image> position and 
try:
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
except:
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    print("Now text in cpu!")
# input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, \
#                                   return_tensors='pt').unsqueeze(0)

# v12/13_t4 MA layer
idx_remove_residual_text   = []#[4] 
idx_restore_residual_text   = []#[31] # this wont work, but we can add criteria to enable the thing.
idx_remove_residual_visual = []#[12, 13] 8, 9, 10, 11, 12, 13
idx_restore_residual_visual   = []#[23] # this wont work, but we can add criteria to enable the thing.
def replace_outlier_with_(act, key='median', threshold=300, topk = 20):
    act_ = act.abs()
    act_restore = []
    mean = torch.mean(act_, dim=(1, 2))
    median = torch.median(act_.flatten(1), dim=-1).values
    zero = torch.zeros_like(mean)
    topk_values, topk_indices = torch.topk(act_.flatten(1), topk)
    # this handles the pytorch 2.1 used in llava do not have unravel_index function
    batch_size, height, width = act.shape  # Assuming `act` is a 3D tensor (B, N, C)
    topk_indices = topk_indices % (height * width)
    topk_row = topk_indices // width  # Compute the row index
    topk_col = topk_indices % width  # Compute the column index
    for bs, (value, indice) in enumerate(zip(topk_values, zip(topk_row, topk_col))):
        for v, idx in zip(value, zip(*indice)):
            if v > 5:
                act_restore.append((v, idx, bs))
                if key == 'median':
                    act[bs][idx] = median[bs]
                elif key == 'mean':
                    act[bs][idx] = mean[bs]
                else:
                    act[bs][idx] = zero[bs]
    return act_restore

def restore_outlier_with_(act, act_restore, key='median', threshold=300, topk = 30):
    for arestore in act_restore:
        v, idx, bs = arestore
        act[bs][idx] = v

def replace_outlier_with_CO(act, key='median', threshold=300, topk = 30):
    act_mean = act.abs().mean(dim=-1)
    act_median = torch.median(act.abs(), dim=-1).values
    act_zero = torch.zeros_like(act_mean)
    # for c_dim in [139, 196, 211, 250, 350, 437, 468, 469, 499, 953]: # vision
    # for c_dim in [61, 110, 282, 371, 550, 599, 893, 973, 1160, 1185, 1375, 1387, 1419, 1812, 1832, 1951, 2076, 2490, 2741, 2816, 2881, 2942, 2983, 3022, 3050, 3154, 3159, 3186, 3272, 3405, 3514, 3526, 3548, 3607, 3837, 3849, 4054, 4490, 4579, 4743, 4899, 4994, 5046]:
    #     act[...,c_dim] = act_zero
    for c_dim in [542]:
        act[...,c_dim] = act_zero


def visionblock_custom_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                causal_attention_mask: torch.Tensor,
                output_attentions: Optional[bool] = False,
            ) -> Tuple[torch.FloatTensor]:
        batchid=0
        # if self.mode == 'replace':
        #     print(f"x1  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        #     # replace_outlier_with_CO(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
        #     self.act += replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
        #     print(f"x1  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}, {len(self.act)}")
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm1 x2
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm2 y1
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm2 y2
        hidden_states = self.mlp(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of mlp y5
        # print(f"y5  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the output of mlp y5
        # print(f"y5  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        if self.mode == 'replace':
            print(f"y5  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
            replace_outlier_with_CO(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
            # self.act += replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
            print(f"y5  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        hidden_states = residual + hidden_states
        # if self.mode == 'restore':
        #     print(f"nextlayer x1 before restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        #     restore_outlier_with_(hidden_states, set(self.act)) # here the input of norm1 x1
        #     print(f"nextlayer x1 before restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()},{len(set(self.act))}")
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

def enable_vision_custom_block(block, block_id, mode='replace'):
    block.block_id = block_id
    import types
    block.mode = mode
    block.forward = types.MethodType(visionblock_custom_forward, block)

def llamablock_custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batchid=0
        # if self.mode == 'replace':
        #     print(f"x1  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        #     # replace_outlier_with_CO(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
        #     self.act += replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
        #     print(f"x1  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}, {len(self.act)}")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm1 x2
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm2 y1
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm2 y2
        hidden_states = self.mlp(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of mlp y8
        # print(f"y8  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the output of mlp y8
        # print(f"y8  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")

        if self.mode == 'replace':
            print(f"y8  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
            replace_outlier_with_CO(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
            # self.act += replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
            print(f"y8  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        hidden_states = residual + hidden_states
        # if self.mode == 'restore':
        #     print(f"nextlayer x1 before restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        #     restore_outlier_with_(hidden_states, set(self.act)) # here the input of norm1 x1
        #     print(f"nextlayer x1 before restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}, {len(set(self.act))}")
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



def enable_text_custom_block(block, block_id, mode='replace'):
    block.block_id = block_id
    import types
    block.mode = mode
    block.forward = types.MethodType(llamablock_custom_forward, block)

def register_custom_block_hooks(keys_to_mitigate=None):
    block_fusion = model.model
    block_vision = model.model.vision_tower.vision_tower.vision_model.encoder
    if len(idx_restore_residual_text)>0:
        block_fusion.act_restore = []
    if len(idx_restore_residual_visual)>0:
        block_vision.act_restore = []
    for idx, block in enumerate(model.model.vision_tower.vision_tower.vision_model.encoder.layers):
        if idx in idx_remove_residual_visual:
            if len(idx_restore_residual_visual)>0:
                block.act = block_vision.act_restore
            enable_vision_custom_block(block, idx)
            print(f'set up replace in vision layer {idx}')
        if idx in idx_restore_residual_visual:
            block.act = block_vision.act_restore
            enable_vision_custom_block(block, idx, mode='restore')
            print(f'set up restore in vision layer {idx}')
    for idx, block in enumerate(model.model.layers):
        if idx in idx_remove_residual_text:
           if len(idx_restore_residual_text)>0:
               block.act = block_fusion.act_restore
           enable_text_custom_block(block, idx)
           print(f'set up replace in fusion layer {idx}')
        if idx in idx_restore_residual_text:
            block.act = block_fusion.act_restore
            enable_text_custom_block(block, idx, mode='restore')
            print(f'set up restore in fusion layer {idx}')


import logging
activations = dict()

def save_activationio(name, in_key=None, out_key=''):
    def hook(model, input, output):
        if in_key is not None:
            activations[name+in_key]=input[0].detach().clone().cpu()
        activations[name+out_key]=output.detach().clone().cpu()
    return hook

def save_attnactivation(name):
    def hook(model, input, output):
        activations[name + '_inputq']=input[0].detach().clone().cpu()
        activations[name + '_inputk']=input[1].detach().clone().cpu()
        activations[name + '_inputv']=input[2].detach().clone().cpu()
        activations[name]=output[0].detach().clone().cpu()
    return hook

# model.vision_tower.vision_tower.vision_model.encoder.layers
# model.layers.self_attn
# model.layers.mlp
def register_detailed_hooks(keys_to_include=None):
    for idx, block in enumerate(model.model.vision_tower.vision_tower.vision_model.encoder.layers):
        idx = None
        if idx is not None:
            ###### ln_1 in and out
            # x_1 and x_2 in fig2-b
            block.layer_norm1.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.norm1",'_in','_out'))


            ###### attn module and require weight for cal hidden activations
            # x_3, x_4, x_5, x_8, x_9
            # block.self_attn.q_proj.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.attn", out_key='q_proj'))
            # block.self_attn.k_proj.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.attn", out_key='k_proj'))
            # block.self_attn.v_proj.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.attn", out_key='v_proj'))
            block.self_attn.out_proj.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.attn", in_key='_score', out_key='out_proj'))

            ###### FFN ln_2 in and out
            block.layer_norm2.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.norm2",'_in','_out'))
            
            ###### FFN Gelu and mlp
            # block.mlp.fc1.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.fc1"))
            block.mlp.fc2.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.fc2"))
            block.mlp.activation_fn.register_forward_hook(save_activationio(f"llava.visual.block_{idx}.gelu"))
            # ['norm1_in', 'norm1_out', 'attnq_proj', 'attnk_proj', 'attnv_proj', 'attn_score', 'attnout_proj',\
            #    'norm2_in', 'norm2_out', 'fc1', 'gelu', 'fc2']
    # if 'mm'
    print("Setting up hook for llava vision model")

    for idx, block in enumerate(model.model.layers):
        
        if idx is not None:
            ###### rms_1 in and out
            # x_1 and x_2 in fig2-b
            block.input_layernorm.register_forward_hook(save_activationio(f"llava.text.block_{idx}.norm1",'_in','_out'))


            ###### attn module and require weight for cal hidden activations
            # x_3, x_4, x_5, x_8, x_9
            # block.self_attn.q_proj.register_forward_hook(save_activationio(f"llava.text.block_{idx}.attn", out_key='q_proj'))
            # block.self_attn.k_proj.register_forward_hook(save_activationio(f"llava.text.block_{idx}.attn", out_key='k_proj'))
            # block.self_attn.v_proj.register_forward_hook(save_activationio(f"llava.text.block_{idx}.attn", out_key='v_proj'))
            block.self_attn.o_proj.register_forward_hook(save_activationio(f"llava.text.block_{idx}.attn", in_key='_score', out_key='out_proj'))


            ###### FFN rms_2 in and out
            # y1, y2
            block.post_attention_layernorm.register_forward_hook(save_activationio(f"llava.text.block_{idx}.norm2",'_in','_out'))
            
            ###### FFN Gelu and mlp
            # y3 -f11, y4 -fc1, y5 -silu, y6, y7
            # block.mlp.gate_proj.register_forward_hook(save_activationio(f"llava.text.block_{idx}.fc11"))
            # block.mlp.up_proj.register_forward_hook(save_activationio(f"llava.text.block_{idx}.fc1"))
            # block.mlp.act_fn.register_forward_hook(save_activationio(f"llava.text.block_{idx}.silu"))
            block.mlp.down_proj.register_forward_hook(save_activationio(f"llava.text.block_{idx}.fc2",'_in','_out'))
            # ['norm1_in', 'norm1_out', 'attnq_proj', 'attnk_proj', 'attnv_proj', 'attn_score', 'attnout_proj',\
            #    'norm2_in', 'norm2_out', 'fc11', 'fc1', 'silu', 'fc2_in', 'fc2_out']
    print("Setting up hook for llava text model")
    # see if this works
    block = model.model.mm_projector
    block[0].register_forward_hook(save_activationio(f"llava.mmproj.fc1"))
    block[1].register_forward_hook(save_activationio(f"llava.mmproj.gelu"))
    block[2].register_forward_hook(save_activationio(f"llava.mmproj.fc2"))
    print("Setting up hook for llava proj model")
    block = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    block.register_forward_hook(save_activationio(f"llava.vision.preln",'_in','_out'))
    print("Setting up hook for llava vision model preln")
register_custom_block_hooks()
register_detailed_hooks()
print("Setting up hook for llava model")


def replace_parameter_CO_Woutlm(model, alayeridx, mlayeridx, channels=[542]):
    layers = model.model.layers
    for c in channels:
        for idx, layer in enumerate(layers):
            if idx in alayeridx:
                W_out = layer.self_attn.o_proj
                weight = W_out.weight.data
                weight_mean = weight.mean(0)
                weight[c] = weight_mean
                W_out.weight.data = weight
            if idx in mlayeridx:
                W_out = layer.mlp.down_proj
                weight = W_out.weight.data
                weight_mean = weight.mean(0)
                weight[c] = weight_mean
                W_out.weight.data = weight
        print(f"finish weight replace in lm selfattn oproj layers{alayeridx}- channel {c}")
        print(f"finish weight replace in lm mlp down layers{mlayeridx}- channel {c}")

def replace_parameter_CO_Wout(model, alayeridx, mlayeridx, channels=[542]):
    for c in channels:
        layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
        for idx, layer in enumerate(layers):
            if idx in alayeridx:
                W_out = layer.self_attn.out_proj
                weight = W_out.weight.data
                weight_mean = weight.mean(0)
                # weight[542] = weight_mean # maybe try mean(0)?
                weight[c] = torch.zeros_like(weight_mean)
                W_out.weight.data = weight
                bias = W_out.bias.data
                bias_mean = bias.mean()
                bias[c] = bias_mean
                W_out.bias.data = bias
            if idx in mlayeridx:
                W_out = layer.mlp.fc2
                weight = W_out.weight.data
                weight_mean = weight.mean(0)
                # weight[542] = weight_mean # maybe try mean(0)?
                weight[c] = torch.zeros_like(weight_mean)
                W_out.weight.data = weight
                bias = W_out.bias.data
                bias_mean = bias.mean()
                bias[c] = bias_mean
                W_out.bias.data = bias
        print(f"finish w/b replace in vit selfattn oproj layers{alayeridx}- channel {c}")
        print(f"finish w/b replace in vit mlp down layers{mlayeridx}- channel {c}")
# attn_layers = [8, 9, 10, 11, 12, 13]
attn_layers = [_ for _ in range(0, 24)]
# mlp_layer = [8, 9, 10, 11, 12, 13]
mlp_layer = [_ for _ in range(0, 24)]
channels = [542, 121, 407, 711]
# replace_parameter_CO_Wout(model, attn_layers, mlp_layer, channels)

attn_layers = [_ for _ in range(0, 40)]
# mlp_layer = [8, 9, 10, 11, 12, 13]
mlp_layer = [_ for _ in range(0, 40)]
channels = [3050 ,4727 ,4743 ,2772 ,1843]
replace_parameter_CO_Woutlm(model, attn_layers, mlp_layer, channels)

with torch.inference_mode():
    (inputs, position_ids,attention_mask, _,inputs_embeds,_) = \
        model.prepare_inputs_labels_for_multimodal(input_ids, None, None, None, None, images, image_sizes)
    inputs = model.prepare_inputs_for_generation(inputs, None, inputs_embeds)
    out = model(**inputs)


# base_dir  ='/scratch/yw6594/cf/vlm/LLaVA/playground/data/eval/scienceqa/actCO_Wout/vit'
base_dir  ='/scratch/yw6594/cf/vlm/LLaVA/playground/data/eval/scienceqa/actCO_Wout/lm'
os.makedirs(base_dir, exist_ok=True)

# torch.save(activations, '/scratch/yw6594/cf/vlm/LLaVA/playground/data/eval/scienceqa/actwithoutMA/y5_v1112_t3/top20' + "/batch2_mm_activations.pt")
# torch.save(activations, '/scratch/yw6594/cf/vlm/LLaVA/playground/data/eval/scienceqa/actwithoutMA/x1_v13_t4/top30' + "/batch2_mm_activations.pt")
# torch.save(activations, '/scratch/yw6594/cf/vlm/LLaVA/playground/data/eval/scienceqa/act' + "/preln_batch2_mm_activations.pt")
# torch.save(activations, '/scratch/yw6594/cf/vlm/LLaVA/playground/data/eval/scienceqa/actwithoutCOfusion' + "/preln_batch2_mm_activations_zero.pt")
# torch.save(activations, base_dir + "/preln_batch2_mm_activations_wout_replace_samlp_wb_024_channels.pt")
torch.save(activations, base_dir + "/preln_batch2_mm_activations_wout_replace_samlp_wb_040_channels.pt")
print(f"Saving training activations  of question {qidx}, break")
####################### y5
######### top3
# y5  before change max at batch0: 61.23799514770508
# y5  after change max at batch0: 22.6053466796875
# y5  before change max at batch0: 174.34912109375
# y5  after change max at batch0: 24.73068618774414
# y8  before change max at batch0: 1388.0
# y8  after change max at batch0: 67.5

######### top10
# y5  before change max at batch0: 61.23799514770508
# y5  after change max at batch0: 11.92331314086914
# y5  before change max at batch0: 191.80125427246094
# y5  after change max at batch0: 16.257169723510742
# y8  before change max at batch0: 1388.0
# y8  after change max at batch0: 36.375
######### top20
# y5  before change max at batch0: 61.23815155029297
# y5  after change max at batch0: 5.27302360534668
# y5  before change max at batch0: 205.81744384765625
# y5  after change max at batch0: 6.2144365310668945
# y8  before change max at batch0: 1389.0
# y8  after change max at batch0: 24.0
######### top30
# y5  before change max at batch0: 61.23815155029297
# y5  after change max at batch0: 4.999103546142578
# y5  before change max at batch0: 206.39056396484375
# y5  after change max at batch0: 4.3100996017456055
# y8  before change max at batch0: 1389.0
# y8  after change max at batch0: 12.8046875

######### top50 # only test till this.
# y5  before change max at batch0: 61.23799514770508
# y5  after change max at batch0: 4.999081611633301
# y5  before change max at batch0: 206.3905029296875
# y5  after change max at batch0: 4.310372352600098
# y8  before change max at batch0: 1388.0
# y8  after change max at batch0: 4.92578125

######### top100
# y5  before change max at batch0: 61.23799514770508
# y5  after change max at batch0: 4.999081611633301
# y5  before change max at batch0: 206.3905029296875
# y5  after change max at batch0: 4.310372352600098
# y8  before change max at batch0: 1388.0
# y8  after change max at batch0: 4.92578125

####################### x1
######### top10
# x1  before change max at batch0: 193.82220458984375
# x1  after change max at batch0: 23.666765213012695
# x1  before change max at batch0: 1394.0
# x1  after change max at batch0: 30.8125

######### top20
# x1  before change max at batch0: 193.82244873046875
# x1  after change max at batch0: 11.835888862609863
# x1  before change max at batch0: 1395.0
# x1  after change max at batch0: 16.40625

######### top30
# x1  before change max at batch0: 193.82244873046875
# x1  after change max at batch0: 5.63663387298584
# x1  before change max at batch0: 1395.0
# x1  after change max at batch0: 7.4453125

######### top50 
# x1  before change max at batch0: 193.82220458984375
# x1  after change max at batch0: 4.9893293380737305
# x1  before change max at batch0: 1394.0
# x1  after change max at batch0: 5.43359375

######### top100 # test this for text part
# x1  before change max at batch0: 193.82220458984375
# x1  after change max at batch0: 4.9893293380737305
# x1  before change max at batch0: 1394.0
# x1  after change max at batch0: 4.99609375