import jax.numpy as jnp
import jax.nn as jnn
import jax
import jax.lax as lax

def random_multinomial_sample_from_logits(rand_logits, prng_key):
    logits = rand_logits
    # 保存原始形状
    probs_shape = None
    if len(logits.shape) >= 3:
        probs_shape = logits.shape
        # 展平为 2D: (batch_size * seq_len, vocab_size)
        logits = logits.reshape(-1, logits.shape[-1])

    # Top-k=1 采样
    topk_logits = jnp.max(logits, axis=-1, keepdims=True)  # 最大 logits
    topk_cls_indices = jnp.argmax(logits, axis=-1, keepdims=True)  # 最大索引
    # 创建全为 -inf 的 logits，仅保留 top-1
    logits = jnp.full_like(logits, -jnp.inf).at[..., topk_cls_indices].set(topk_logits)

    # Softmax 转换为概率
    probs = jnn.softmax(logits, axis=-1)

    # 采样 token（尽管 top-k=1 使采样确定性）
    rand_tokens = jax.random.categorical(prng_key, logits, axis=-1)

    # 恢复形状
    if probs_shape is not None:
        rand_tokens = rand_tokens.reshape(probs_shape[:-1])
        probs = probs.reshape(probs_shape)

    return rand_tokens, probs

# Prefill
#NOTE: SJD
def get_multi_token_for_preparation(
    rand_token_num, #窗口大小
    acceptance_length,
    input_ids,
    input_probs,
    candidate_ids, 
    candidate_probs, 
    input_ids_len,
    img_width = 16,
    multi_token_init_scheme=None, #初始化方案, horizon or vertical
    prefill_num = 0,#prompt
    eps = 1e-7, # igorne it
):
    # 1.1 Init more, change the prefill_num
    pad_len = 0 # pad_len = right_above
    img_width = img_width + pad_len if img_width is not None else 0
    if (img_width > 0) and (rand_token_num > 0):
        positon_indices = jnp.arange(0, candidate_ids.shape[-1]+rand_token_num*2, dtype=jnp.int32)#BUG
        positon_indices = jax.lax.dynamic_slice(positon_indices, (input_ids_len,), (rand_token_num,)) - prefill_num
        horizon_indices = positon_indices % img_width
        vertical_indices = positon_indices // img_width

        # inition plan only horizon now
        # horizon case
        if 'horizon' in multi_token_init_scheme:
            last_vertical_indices = vertical_indices
            last_horizon_indices = horizon_indices - rand_token_num
        elif 'vertical' in multi_token_init_scheme:
            last_vertical_indices = vertical_indices - 1
            last_horizon_indices = horizon_indices
        elif 'random' in multi_token_init_scheme:
            last_vertical_indices = vertical_indices
            last_horizon_indices = horizon_indices
        else:
            assert False, f"multi_token_init_scheme should be 'horizon' 'random' or 'vertical', but got {multi_token_init_scheme}"

        last_candidate_tokens = candidate_ids
        last_candidate_probs = candidate_probs

        # 一维的索引值 可能是空的, 可能是非空的. 如果是非空的, 就采取某种策略去做token的选取
        # last_vertical_indices and last_horizon_indices is (2, rand_token_num)
        last_flatten_indices = last_vertical_indices * img_width + last_horizon_indices + prefill_num
        # e.g., last indices [100, 101, 102], but the current indices up to 100, 
        # and 101, 102 depends on the values from 100 (but 100 has not been appended to input-ids yet)
        last_flatten_indices = jnp.clip(last_flatten_indices, a_min=0, a_max=last_candidate_tokens.shape[1] - 1)
        
        # Repeat的token以及logits last_input_tokens-->(2, rand_token_num)
        last_resampled_input_tokens = last_candidate_tokens[:, last_flatten_indices]
        last_resampled_input_probs = last_candidate_probs[:, last_flatten_indices]
        last_resampled_input_logits = jnp.log(last_resampled_input_probs.astype(jnp.float32) + eps)

        # Transform plan
        if 'sample' in multi_token_init_scheme:
            resampled_rand_tokens, resampled_rand_probs = random_multinomial_sample_from_logits(
                last_resampled_input_logits
            ) # TODO: jax format need to change
            rand_tokens = resampled_rand_tokens
            rand_probs = resampled_rand_probs
        elif 'repeat' in multi_token_init_scheme:
            rand_tokens = last_resampled_input_tokens
            rand_probs = last_resampled_input_probs
        else:
            assert False, f"multi_token_init_scheme should be 'sample' or 'repeat', but got {multi_token_init_scheme}"
        def prefill_token_line(i, A):#做替换填充 # BUG
            input_ids_i, input_probs_i = A
            input_ids_i = input_ids_i.at[:, i].set(rand_tokens[:,i])
            input_probs_i = input_probs_i.at[:, i].set(rand_probs[:,i])
            return input_ids_i, input_probs_i
        input_ids, input_probs = jax.lax.fori_loop(
            -(acceptance_length - 1), 0, prefill_token_line, (input_ids, input_probs)
        )
    
    return input_ids, input_probs

#NOTE: FSJD
def get_update_window_token_FSJD(
    rand_token_num, #窗口大小
    acceptance_length,
    input_ids,
    input_probs,
    candidate_ids, 
    candidate_probs, 
    input_ids_len,
    img_width = 16,
    multi_token_init_scheme=None, #初始化方案, horizon or vertical
    prefill_num = 0,#prompt
    eps = 1e-7, # igorne it
):
    # 1.1 Init more, change the prefill_num
    pad_len = 0 # pad_len = right_above
    img_width = img_width + pad_len if img_width is not None else 0
    # 2. if there exist input_ids to cat
    if (img_width > 0) and (rand_token_num > 0):
        positon_indices = jnp.arange(0, candidate_ids.shape[-1]+rand_token_num*2, dtype=jnp.int32)#BUG
        positon_indices = jax.lax.dynamic_slice(positon_indices, (input_ids_len,), (rand_token_num,)) - prefill_num
        horizon_indices = positon_indices % img_width
        vertical_indices = positon_indices // img_width

        # inition plan only horizon now
        # horizon case
        if 'horizon' in multi_token_init_scheme:
            last_vertical_indices = vertical_indices
            last_horizon_indices = horizon_indices - rand_token_num
        elif 'vertical' in multi_token_init_scheme:
            last_vertical_indices = vertical_indices - 1
            last_horizon_indices = horizon_indices
        elif 'random' in multi_token_init_scheme:
            last_vertical_indices = vertical_indices
            last_horizon_indices = horizon_indices
        else:
            assert False, f"multi_token_init_scheme should be 'horizon' 'random' or 'vertical', but got {multi_token_init_scheme}"
        
        # TODO: 似乎这样的更新存在一定的问题
        last_candidate_tokens = candidate_ids
        last_candidate_probs = candidate_probs

        # TODO: 不知道索引的对不对, 对的
        last_flatten_indices = last_vertical_indices * img_width + last_horizon_indices + prefill_num
        # e.g., last indices [100, 101, 102], but the current indices up to 100, 
        # and 101, 102 depends on the values from 100 (but 100 has not been appended to input-ids yet)
        last_frame_flatten_indices = last_flatten_indices - img_width * img_width - 1 - prefill_num# 1 is 8196 pad token
        last_flatten_indices = jax.lax.cond(
            last_frame_flatten_indices.min() < 0,
            lambda: jnp.clip(last_flatten_indices, a_min=0, a_max=last_candidate_tokens.shape[1] - 1),
            lambda: last_frame_flatten_indices + prefill_num,
        )
        
        # Repeat的token以及logits last_input_tokens-->(2, rand_token_num)
        last_resampled_input_tokens = last_candidate_tokens[:, last_flatten_indices]
        last_resampled_input_probs = last_candidate_probs[:, last_flatten_indices]
        last_resampled_input_logits = jnp.log(last_resampled_input_probs.astype(jnp.float32) + eps)

        # Transform plan
        if 'sample' in multi_token_init_scheme:
            resampled_rand_tokens, resampled_rand_probs = random_multinomial_sample_from_logits(
                last_resampled_input_logits
            ) # TODO: jax format need to change
            rand_tokens = resampled_rand_tokens
            rand_probs = resampled_rand_probs
        elif 'repeat' in multi_token_init_scheme:
            rand_tokens = last_resampled_input_tokens
            rand_probs = last_resampled_input_probs
        else:
            assert False, f"multi_token_init_scheme should be 'sample' or 'repeat', but got {multi_token_init_scheme}"
        def prefill_token_line(i, A):#做替换填充
            # input_ids_i: (2, rand_token_num)
            # last_resampled_input_logits: (2, rand_token_num)
            input_ids_i, input_probs_i = A
            input_ids_i = input_ids_i.at[:, i].set(rand_tokens[:,i])
            input_probs_i = input_probs_i.at[:, i].set(rand_probs[:,i])
            return input_ids_i, input_probs_i
        input_ids, input_probs = jax.lax.fori_loop(
            -(acceptance_length - 1), 0, prefill_token_line, (input_ids, input_probs)
        )
    
    return input_ids, input_probs

# Verify
import jax
import jax.numpy as jnp
from jax import random
from typing import List, Callable, Optional, Tuple

def init_array(len, img_vocab_size):
    token_ids = jax.random.randint(
        key=jax.random.PRNGKey(0),
        shape=(1, len),
        minval=3,
        maxval=img_vocab_size
    )
    token_probs = jnp.zeros((1, len, img_vocab_size), dtype=jnp.float32)
    token_probs = token_probs.at[:, jnp.arange(len), token_ids].set(1.0)

    return token_ids, token_probs

class SpeculativeSampler:
    def __init__(
        self,
        collected_draft_logits: List[jnp.ndarray] = None,
        collected_advanced_logits: List[jnp.ndarray] = None,
        max_num_collected_logits: int = 2,
        generator: Optional[random.PRNGKey] = None,
        draft_type: str = 'jacobian_states',
        sampling_last_draft_token: Optional[jnp.ndarray] = None,
    ):
        # self.max_num_collected_logits = max_num_collected_logits
        # self.collected_draft_logits = collected_draft_logits if collected_draft_logits is not None else []
        # self.collected_advanced_logits = collected_advanced_logits if collected_advanced_logits is not None else []

        self.draft_token_index_selector = lambda x: x
        self.next_token_index_selector = (lambda x: x - 1) if draft_type == 'jacobian_states' else lambda x: x

        self.generator = generator if generator is not None else random.PRNGKey(0)
        # self.image_token_list = jnp.arange(4, 8196)

    def get_reject_sampling_logits(self, token_advanced_prob: jnp.ndarray, token_draft_prob: jnp.ndarray) -> jnp.ndarray:
        """计算拒绝采样的 logits"""
        pos_delta_logits = jnp.log(jnp.maximum(token_advanced_prob - token_draft_prob, 0))
        return pos_delta_logits

    def reject_sampling_single_token(
        self,
        token_next_prob: jnp.ndarray,
        token_draft_prob: jnp.ndarray,
        logits_processor: Optional[Callable] = None,
        logits_warper: Optional[Callable] = None,
        all_collected_input_ids: Optional[jnp.ndarray] = None,
        key: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """对单个 token 进行拒绝采样"""
        pos_delta_logits = self.get_reject_sampling_logits(token_next_prob, token_draft_prob)
        shape_pos_delta_logits = pos_delta_logits.shape

        if logits_processor is not None or logits_warper is not None:
            # 确保输入形状正确
            while len(all_collected_input_ids.shape) < 2:
                all_collected_input_ids = jnp.expand_dims(all_collected_input_ids, axis=0)
            while len(pos_delta_logits.shape) < 2:
                pos_delta_logits = jnp.expand_dims(pos_delta_logits, axis=0)

        if logits_processor is not None and len(logits_processor) != 0:
            pos_delta_logits = logits_processor(all_collected_input_ids, pos_delta_logits, 1)

        if logits_warper is not None:# 1是用来填充的,没啥用
            pos_delta_logits = logits_warper(all_collected_input_ids, pos_delta_logits, cur_len=1)

        pos_delta_logits = pos_delta_logits.reshape(shape_pos_delta_logits)
        probs = jax.nn.softmax(pos_delta_logits, axis=-1)
        resampled_scores = probs

        probs = jnp.atleast_2d(probs)
        key, subkey = random.split(key if key is not None else self.generator)
        resampled_tokens = random.categorical(subkey, jnp.log(probs), axis=-1)
        resampled_tokens = resampled_tokens.squeeze(-1)

        return resampled_tokens, resampled_scores

    def __call__(
        self,
        draft_tokens: jnp.ndarray,
        next_tokens: jnp.ndarray,
        draft_prob: jnp.ndarray,
        next_prob: jnp.ndarray,
        logits_processor: Optional[Callable] = None,
        logits_warper: Optional[Callable] = None,
        all_collected_input_ids: Optional[jnp.ndarray] = None,
        key: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # """主调用函数，执行推测采样"""
        # draft_tokens: [B, L], advanced_tokens: [B, L], draft_prob: [B, L, V], advanced_prob: [B, L, V]
        assert draft_tokens.shape == next_tokens.shape
        assert draft_prob.shape == next_prob.shape
        assert len(draft_tokens.shape) == 2 and len(draft_prob.shape) == 3

        B, L = draft_tokens.shape
        key = key if key is not None else self.generator
        key, subkey = random.split(key)
        rs = random.uniform(subkey, next_prob.shape)
        draft_token_index_selector = self.draft_token_index_selector
        next_token_index_selector = self.next_token_index_selector

        # 三大记录矩阵, 全局变量
        resampled_next_tokens = next_tokens.copy()
        resampled_next_scores = next_prob.copy()
        rejected_index_list = jnp.full((B,L), L, dtype=jnp.int32)

        def process_batch(b, b_carry):
            next_tokens_b, next_probs_b, rejected_index_list_b, key_b = b_carry

            # misaligned_idx 记录着last的拒绝的index, 一开始预期全部接受到最后一个位置
            def line_spective_sample(i, i_carry):
                next_tokens_i, next_probs_i, rejected_index_list_i, key_i = i_carry
                draft_token_index = draft_token_index_selector(i)
                target_token_index = next_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index]
                # 1. 获得当前索引的概率
                sampled_target_prob = next_prob[b, target_token_index, cls_idx]
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]
                r = rs[b, i, cls_idx]
                def accept_fn(key_i):
                    # 2.1 如果接受, last的拒绝的index不变
                    return (
                        next_tokens_i.at[b, target_token_index].set(cls_idx),
                        next_probs_i.at[b, target_token_index].set(draft_prob[b, draft_token_index]),
                        rejected_index_list_i,
                        key_i
                    )
                def reject_fn(key_i):
                    # 此外还有重新采样
                    key_i, subkey = random.split(key_i)
                    # all_collected_input_ids = jnp.concatenate([
                    #         all_collected_input_ids[b],
                    #         next_tokens_i[b, :target_token_index]
                    #     ], axis=-1)
                    all_collected_input_ids = jnp.empty((1,2,0), dtype=jnp.int32) # TODO:all_collected_input_ids 不会被用上, 没必要
                    resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                        token_next_prob=next_prob[b, target_token_index],
                        token_draft_prob=draft_prob[b, draft_token_index],
                        logits_processor=logits_processor,
                        logits_warper=logits_warper,
                        all_collected_input_ids=all_collected_input_ids,
                        key=subkey
                    )

                    return (
                        next_tokens_i.at[b, target_token_index].set(resampled_tokens),
                        next_probs_i.at[b, target_token_index].set(resampled_scores),
                        rejected_index_list_i.at[b,i].set(i),#如果拒绝, last的拒绝的index更新
                        key_i
                    )

                # 2. 进行推测性采样
                next_tokens_i, next_probs_i, \
                rejected_index_list_i, key_i= jax.lax.cond(
                    r < jnp.minimum(sampled_target_prob / sampled_draft_prob, 1.0),#BUG: 可能弄反了
                    accept_fn,
                    reject_fn,
                    key_i
                )

                return next_tokens_i, next_probs_i, rejected_index_list_i, key_i
            
            # rejected_idx是会改变的, 每个point都会改变, 所以要更新
            next_tokens_b, next_probs_b, \
            rejected_index_list_b, key_b = jax.lax.fori_loop(
                1, L, line_spective_sample, (next_tokens_b, next_probs_b, rejected_index_list_b, key_b)
            )
            return next_tokens_b, next_probs_b, rejected_index_list_b, key_b

        # 更新的有: new token,new prob, new rejected index, key 保证了每个Line有同样的随机性, 平行验证
        resampled_next_tokens, resampled_next_scores, rejected_index_list, _ = jax.lax.fori_loop(
            0, B, process_batch, (resampled_next_tokens, resampled_next_scores, rejected_index_list, key)
        )

        # 一定是 [L-1, ...] 全都接受不会改变最小值L-1, 第一个被拒绝的就是那个应该被索引的地方. 为树状结构埋下伏笔 max_rejected_idx=(1,...,L-1), 若都接受, 则为L
        max_rejected_idx = rejected_index_list.min(axis=-1).max(axis=0)
        return max_rejected_idx, resampled_next_tokens, resampled_next_scores

def find_first_misaligned_token_inds(input_tokens, next_tokens):
    b=0
    L = input_tokens.shape[1]
    rejected_index_list = jnp.full((b+1,L), L, dtype=jnp.int32)
    def greedy(i, rejected_index_list_i):
        def accept_fn(list_i):
            return list_i
        def reject_fn(list_i):
            return list_i.at[b,i].set(i)
        rejected_index_list_i= jax.lax.cond(
            input_tokens[b, i] == next_tokens[b, i-1],
            accept_fn,
            reject_fn,
            rejected_index_list_i
        )
        return rejected_index_list_i
    rejected_index_list = jax.lax.fori_loop(
        1, L, greedy, (rejected_index_list)
    )
    reject_index = rejected_index_list.min()
    return reject_index

def prefix_matching_next_tokens(
    input_tokens,
    next_tokens,
    input_probs,
    next_probs,
    prefix_token_sampler=None,
    **kwargs
):
    if prefix_token_sampler is None:
        acceptance_length = find_first_misaligned_token_inds(
            input_tokens, next_tokens
        )
    else:
        acceptance_length, next_tokens, next_probs = prefix_token_sampler(
            input_tokens, next_tokens,
            input_probs, next_probs,
            **kwargs
        )
    
    B,L,D = next_probs.shape
    
    matched_next_tokens = jnp.full((1, L), 0, dtype=jnp.int32)
    matched_next_probs = jnp.full((1, L, D), 0, dtype=jnp.float32)
    unmatched_next_tokens = jnp.full((1, L-1), -1, dtype=jnp.int32)
    unmatched_next_probs = jnp.full((1, L-1, D), -1, dtype=jnp.float32)
    def getslice(i, A):
        matched_tokens, matched_probs, unmatched_tokens, unmatched_probs = A
        def match(A1):
            matched_tokens, matched_probs, unmatched_tokens, unmatched_probs = A1
            matched_tokens = lax.dynamic_update_slice(matched_tokens, next_tokens[:, i][:,None], (0, i))
            matched_probs = lax.dynamic_update_slice(matched_probs, next_probs[:, i][:,None,:], (0, i, 0))
            return matched_tokens, matched_probs, unmatched_tokens, unmatched_probs
        def unmatch(A1):
            matched_tokens, matched_probs, unmatched_tokens, unmatched_probs = A1
            # 空出index=0的位置给最新accept的token在unmatched_tokens, unmatched_probs中
            j = i - acceptance_length #j为不match的index
            if unmatched_tokens.shape[1]>=1:
                unmatched_tokens = lax.dynamic_update_slice(unmatched_tokens, next_tokens[:, i][:,None], (0, j))
                unmatched_probs = lax.dynamic_update_slice(unmatched_probs, next_probs[:, i][:,None,:], (0, j, 0))
            return matched_tokens, matched_probs, unmatched_tokens, unmatched_probs
        matched_tokens, matched_probs, unmatched_tokens, unmatched_probs = jax.lax.cond(
            i < acceptance_length,
            match,
            unmatch,
            (matched_tokens, matched_probs, unmatched_tokens, unmatched_probs)
        )
        return matched_tokens, matched_probs, unmatched_tokens, unmatched_probs
    matched_next_tokens, matched_next_probs, unmatched_next_tokens, unmatched_next_probs = jax.lax.fori_loop(
        0, L, getslice, (matched_next_tokens, matched_next_probs, unmatched_next_tokens, unmatched_next_probs)
    )
    running_tokens = jnp.concatenate([next_tokens[:, acceptance_length-1][:,None],unmatched_next_tokens], axis=1)
    running_probs = jnp.concatenate([next_probs[:, acceptance_length-1][:,None,:],unmatched_next_probs], axis=1)
    return (
        acceptance_length,
        matched_next_tokens,
        running_tokens,
        matched_next_probs,
        running_probs
    )

# For adapt
# Method 1: Get the first 100 pairs
def get_order_pairs(data,len=100):
    return data[:len]

# Method 2: Randomly get 100 pairs
def get_random_pairs(data,len=100):
    return random.sample(data, len)

#NOTE: 为了控制载入不超限

#NOTE: DEBUG

def debug(llama_config,params, layer=32, scan_layers=False,
          max_sequence_length=2048):
    # NOTE:debug
    llama_config.update(dict(
        num_hidden_layers=layer,
        scan_layers=scan_layers
    ))
    
    llama_config.update(dict(
            max_sequence_length=max_sequence_length
        ))
    
    
    #NOTE:debug-->control layer
    from flax.core import freeze, unfreeze
    params = unfreeze(params)
    scan_decoder = params['params']['transformer']['h']['scan_decoder']
    # Trim all parameters in scan_decoder to match num_hidden_layers
    for section in scan_decoder:
        for key in scan_decoder[section]:
            if isinstance(scan_decoder[section][key], dict) and 'kernel' in scan_decoder[section][key]:
                kernel = scan_decoder[section][key]['kernel']
                if kernel.shape[0] > layer:
                    scan_decoder[section][key]['kernel'] = kernel[:layer]
                elif kernel.shape[0] < layer:
                    # Pad with zeros or repeat if necessary (optional, depending on your needs)
                    raise ValueError(f"Parameter {section}/{key}/kernel has unexpected shape {kernel.shape}")
            elif section in ['attention_norm', 'ffn_norm'] and 'kernel' in scan_decoder[section]:
                kernel = scan_decoder[section]['kernel']
                if kernel.shape[0] > layer:
                    scan_decoder[section]['kernel'] = kernel[:layer]
                elif kernel.shape[0] < layer:
                    raise ValueError(f"Parameter {section}/kernel has unexpected shape {kernel.shape}")
    #NOTE:scan_decoder模式取消# 假设 num_hidden_layers=2

    if llama_config.scan_layers:
        params['params']['transformer']['h']['scan_decoder'] = scan_decoder
    else:# 假设 num_hidden_layers=2
        new_h = {}
        for layer_idx in range(layer):
            layer_params = {}
            # 处理 attention 参数
            layer_params['attention'] = {}
            for attn_key in ['wk', 'wo', 'wq', 'wv']:
                if attn_key in scan_decoder['attention']:
                    layer_params['attention'][attn_key] = {
                        'kernel': scan_decoder['attention'][attn_key]['kernel'][layer_idx]
                    }
            # 处理 attention_norm
            if 'attention_norm' in scan_decoder:
                layer_params['attention_norm'] = {
                    'kernel': scan_decoder['attention_norm']['kernel'][layer_idx]
                }
            # 处理 feed_forward 参数
            layer_params['feed_forward'] = {}
            for ffn_key in ['w1', 'w2', 'w3']:
                if ffn_key in scan_decoder['feed_forward']:
                    layer_params['feed_forward'][ffn_key] = {
                        'kernel': scan_decoder['feed_forward'][ffn_key]['kernel'][layer_idx]
                    }
            # 处理 ffn_norm
            if 'ffn_norm' in scan_decoder:
                layer_params['ffn_norm'] = {
                    'kernel': scan_decoder['ffn_norm']['kernel'][layer_idx]
                }
            new_h[str(layer_idx)] = layer_params
        params['params']['transformer']['h'] = new_h
    
    # 转换为 jax.Array
    # params = jax.tree_util.tree_map(jnp.asarray, params)
    params = freeze(params)
    #NOTE:end
    return llama_config, params

def judge_token_sequence(sequences,sequences_label,cur_len,id=0):
    sequences_label = sequences_label[0]
    # if cur_len>128:
    #     if jnp.all(sequences[0,128:cur_len]==sequences_label[id][:(cur_len-128)%257]) \
    #         and jnp.all(sequences[1,128:cur_len]==sequences_label[id][:(cur_len-128)%257]):
    #         return True
    #     return False
    # return True

    def body_fun(carry):
        cur_len, _ = carry
        seq0_match = jnp.all(sequences[0, 128:cur_len] == sequences_label[id][:(cur_len - 128) % 257])
        seq1_match = jnp.all(sequences[1, 128:cur_len] == sequences_label[id][:(cur_len - 128) % 257])
        return (cur_len, seq0_match & seq1_match)

    # Initial carry tuple (cur_len, True)
    carry = (cur_len, True)
    carry = jax.lax.cond(
        cur_len > 128, 
        lambda c: body_fun(c), 
        lambda c: c, carry)
    _, result = carry

    return result