import torch
import jax.numpy as jnp
import jax.nn as jnn
import jax
import torch.nn.functional as F
import jax.lax as lax

def random_multinomial_sample_from_logits(rand_logits, prng_key):
    """
    JAX version of random_multinomial_sample_from_logits.
    
    Args:
        logits: jnp.ndarray, shape (batch_size, seq_len, vocab_size) or (batch_size, vocab_size)
        prng_key: jnp.ndarray, JAX PRNG key for random sampling
    
    Returns:
        rand_tokens: jnp.ndarray, sampled token indices, shape (batch_size, seq_len)
        probs: jnp.ndarray, softmax probabilities, shape (batch_size, seq_len, vocab_size)
    """
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
def get_multi_token_for_preparation(
    rand_token_num, #窗口大小
    input_ids, #已有的token
    temporary_collected_scores, #distribution of token
    img_width = 16,
    multi_token_init_scheme=None, #初始化方案, horizon or vertical
    prefill_num = 0,#prompt
    last_input_tokens=None, #待拼接token, 可能是最新的那一个token, 是上次的没有match到token
    last_input_scores=None, # the same
    eps = 1e-7, # igorne it
    additional_tokens_len = 0,
):
    # if multi_token_init_scheme != 'random':
    # 1. Init rand_tokens是candidate token. 不知道为什么要这样做, 看起来可以删除, 重新写初始化
    img_vocab_size = 8448
    rand_tokens = jax.random.randint(
        key=jax.random.PRNGKey(0),
        shape=(1, rand_token_num),
        minval=3,
        maxval=img_vocab_size
    )
    rand_tokens_scores = jnp.zeros((1, rand_token_num, 8448), dtype=jnp.float32)
    rand_tokens_scores = rand_tokens_scores.at[:, jnp.arange(rand_tokens.shape[-1]), rand_tokens].set(1.0)

    rand_tokens = jnp.tile(rand_tokens, (2, 1))
    
    if multi_token_init_scheme in ["random"]:
        return rand_tokens, rand_tokens_scores
    else:
        # 1.1 Init more, change the prefill_num
        pad_len = 0 # pad_len = right_above
        img_width = img_width + pad_len if img_width is not None else 0
        input_ids_len = input_ids.shape[1]
        
        # 2. if there exist input_ids to cat
        if (img_width > 0) and (input_ids_len + additional_tokens_len >= prefill_num) and (rand_token_num > 0):
            
            # TODO: check the indices
            horizon_indices = (jnp.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                dtype=jnp.int32
            ) - prefill_num) % img_width
            vertical_indices = (jnp.arange(
                input_ids_len + additional_tokens_len, 
                input_ids_len + additional_tokens_len + rand_token_num, 
                dtype=jnp.int32
            ) - prefill_num) // img_width

            # inition plan only horizon now
            # horizon case
            if 'horizon' in multi_token_init_scheme:
                valid_indices = (horizon_indices - 1 >= 0)
                last_vertical_indices = vertical_indices
                last_horizon_indices = horizon_indices - rand_token_num
            elif 'vertical' in multi_token_init_scheme:
                valid_indices = (vertical_indices - 1 >= 0)
                last_vertical_indices = vertical_indices - 1
                last_horizon_indices = horizon_indices
            else:
                assert False, f"multi_token_init_scheme should be 'horizon' or 'vertical', but got {multi_token_init_scheme}"
            
            # 拼接操作--拼接上次剩下的
            last_input_tokens = jnp.concatenate(
                [input_ids, last_input_tokens], axis=1
                ) if last_input_tokens is not None else input_ids
            last_input_scores = jnp.concatenate(
                [temporary_collected_scores, last_input_scores], axis=1
                ) if last_input_scores is not None else temporary_collected_scores
            last_input_logits = jnp.log(last_input_scores.astype(jnp.float32) + eps)

            # 一维的索引值
            last_flatten_indices = last_vertical_indices[valid_indices] * img_width + last_horizon_indices[valid_indices] + prefill_num
            
            # e.g., last indices [100, 101, 102], but the current indices up to 100, 
            # and 101, 102 depends on the values from 100 (but 100 has not been appended to input-ids yet)
            last_flatten_indices = jnp.clip(last_flatten_indices, a_min=0, a_max=last_input_tokens.shape[1] - 1)
            
            # repeat 的token以及logits last_input_tokens-->(2, len)
            last_resampled_input_tokens = last_input_tokens[:, last_flatten_indices]
            last_resampled_input_logits = last_input_logits[:, last_flatten_indices]

            # Transform plan
            # 将布尔索引转换为整数索引, 在rand的里面
            valid_indices_int = jnp.where(valid_indices)[0]  # 提取 True 对应的索引
            if 'sample' in multi_token_init_scheme:
                resampled_rand_tokens, resampled_scores_of_rand_tokens = random_multinomial_sample_from_logits(
                    last_resampled_input_logits
                ) # TODO: jax format need to change
                rand_tokens_scores = rand_tokens_scores.at[:, valid_indices_int].set(0.0)#重新设置索引值
                if valid_indices_int.shape[-1]>0:
                    # 执行 scatter 操作 如果repeat成功的话
                    rand_tokens_scores = rand_tokens_scores.at[:, valid_indices_int, resampled_rand_tokens[0]].set(1.0)
                else:
                    # 如果 valid_indices 全为 False，返回原始张量
                    pass
            elif 'repeat' in multi_token_init_scheme:
                resampled_rand_tokens = last_resampled_input_tokens
                rand_tokens_scores = rand_tokens_scores.at[:, valid_indices_int].set(0.0)#重新设置索引值
                if valid_indices_int.shape[-1]>0:
                    # 执行 scatter 操作 如果repeat成功的话
                    rand_tokens_scores = rand_tokens_scores.at[:, valid_indices_int, resampled_rand_tokens[0]].set(1.0)
                else:
                    # 如果 valid_indices 全为 False，返回原始张量
                    pass
            else:
                assert False, f"multi_token_init_scheme should be 'sample' or 'repeat', but got {multi_token_init_scheme}"
            # rand_tokens-->(2, len)
            rand_tokens = rand_tokens.at[:, valid_indices_int].set(resampled_rand_tokens)
        
        return rand_tokens, rand_tokens_scores

# Verify
import jax
import jax.numpy as jnp
from jax import random
from typing import List, Callable, Optional, Tuple
import functools

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
        self.max_num_collected_logits = max_num_collected_logits
        self.collected_draft_logits = collected_draft_logits if collected_draft_logits is not None else []
        self.collected_advanced_logits = collected_advanced_logits if collected_advanced_logits is not None else []

        self.draft_token_index_selector = lambda x: x
        self.next_token_index_selector = (lambda x: x - 1) if draft_type == 'jacobian_states' else lambda x: x

        self.generator = generator if generator is not None else random.PRNGKey(0)
        self.image_token_list = jnp.arange(4, 8196)

        # 初始化拒绝采样参数
        self.sampling_last_draft_token = sampling_last_draft_token if sampling_last_draft_token is not None else jnp.array([0])

    def get_reject_sampling_logits(self, token_advanced_prob: jnp.ndarray, token_draft_prob: jnp.ndarray) -> jnp.ndarray:
        """计算拒绝采样的 logits"""
        pos_delta_logits = jnp.log(jnp.maximum(token_advanced_prob - token_draft_prob, 0))
        return pos_delta_logits

    def reject_sampling_single_token(
        self,
        token_advanced_prob: jnp.ndarray,
        token_draft_prob: jnp.ndarray,
        logits_processor: Optional[Callable] = None,
        logits_warper: Optional[Callable] = None,
        all_collected_input_ids: Optional[jnp.ndarray] = None,
        key: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """对单个 token 进行拒绝采样"""
        pos_delta_logits = self.get_reject_sampling_logits(token_advanced_prob, token_draft_prob)
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

        resampled_next_tokens = next_tokens.copy()
        resampled_next_scores = next_prob.copy()
        first_misaligned_token_inds = jnp.full((B,), L, dtype=jnp.int32)

        def process_batch(b, b_carry):
            b_next_tokens, b_next_scores, misaligned_inds, key = b_carry
            draft_token_index_selector = self.draft_token_index_selector
            next_token_index_selector = self.next_token_index_selector

            def scan_body(i, loop_carry):
                next_tokens_i, next_scores_i, misaligned_idx, key_i, reject_flag = loop_carry
                if reject_flag: #break 语句
                    return loop_carry
                draft_token_index = draft_token_index_selector(i)
                target_token_index = next_token_index_selector(i)
                cls_idx = draft_tokens[b, draft_token_index]

                sampled_target_prob = next_prob[b, target_token_index, cls_idx]
                sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]
                r = rs[b, i, cls_idx]

                self.sampling_last_draft_token = self.sampling_last_draft_token.at[b].set(cls_idx)

                def accept_fn(key_i):
                    return (
                        next_tokens_i.at[b, target_token_index].set(cls_idx),
                        next_scores_i.at[b, target_token_index].set(draft_prob[b, draft_token_index]),
                        misaligned_idx,
                        key_i,
                        False
                    )

                def reject_fn(key_i):
                    key_i, subkey = random.split(key_i)

                    resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                        token_advanced_prob=next_prob[b, target_token_index],
                        token_draft_prob=draft_prob[b, draft_token_index],
                        logits_processor=logits_processor,
                        logits_warper=logits_warper,
                        all_collected_input_ids=jnp.concatenate([
                            all_collected_input_ids[b],
                            next_tokens_i[b, :target_token_index]
                        ], axis=-1),
                        key=subkey
                    )

                    return (#这里理论上不该改变target的分布的, 但是根据公式, 其要改变.
                        next_tokens_i.at[b, target_token_index].set(resampled_tokens),
                        next_scores_i.at[b, target_token_index].set(resampled_scores),
                        i,
                        key_i,
                        True
                    )

                return jax.lax.cond(
                    r < jnp.minimum(sampled_target_prob / sampled_draft_prob, 1.0),
                    accept_fn,
                    reject_fn,
                    key_i
                )

            final_tokens, final_scores, misaligned_idx, new_key, _ = jax.lax.fori_loop(
                1, L, scan_body, (b_next_tokens, b_next_scores, L, key, False)
            )

            return final_tokens, final_scores, misaligned_inds.at[b].set(misaligned_idx), new_key

        resampled_next_tokens, resampled_next_scores, first_misaligned_token_inds, _ = jax.lax.fori_loop(
            0, B, process_batch, (resampled_next_tokens, resampled_next_scores, first_misaligned_token_inds, key)
        )

        misaligned_idx = first_misaligned_token_inds.min()
        return misaligned_idx-1, resampled_next_tokens, resampled_next_scores

def find_first_misaligned_token_inds(input_tokens, next_tokens):
    b=0
    accept_index = 0 # accept_index in next_tokens
    for i in range(1, input_tokens.shape[0]):
        if input_tokens[b, i] == next_tokens[b, i-1]:
            accept_index = i
        else:
            pass
    return accept_index

def prefix_matching_next_tokens(
    input_tokens,
    next_tokens,
    input_probs,
    next_probs,
    prefix_token_sampler=None,
    **kwargs
):
    """
    Perform prefix matching between model input IDs and next tokens.
    
    Args:
        model_input_ids: JAX array of shape [B, L], input token IDs.
        next_tokens: JAX array of shape [B, L], next token IDs.
        next_token_scores: JAX array of shape [B, L], scores for next tokens.
        input_token_scores: JAX array of shape [B, L] or None, scores for input tokens.
        prefix_token_sampler: Callable or None, sampler for prefix tokens.
        **kwargs: Additional arguments passed to prefix_token_sampler.
    
    Returns:
        Tuple of:
        - matched_num: Integer, number of matched tokens.
        - matched_next_tokens: JAX array, matched next tokens.
        - unmatched_next_tokens: JAX array, unmatched next tokens.
        - matched_next_scores: JAX array, scores for matched tokens.
        - unmatched_next_scores: JAX array, scores for unmatched tokens.
    """
    def default_path():
        """Handle case when prefix_token_sampler is None."""
        match_index = find_first_misaligned_token_inds(
            input_tokens, next_tokens
        )
        return match_index, next_tokens, next_probs
    
    def sampler_path(input_tokens, next_tokens, input_probs, next_token_scores):
        """Handle case when prefix_token_sampler is provided."""
        match_index, next_tokens, next_token_scores = prefix_token_sampler(
            draft_tokens=input_tokens,
            next_tokens=next_tokens,
            draft_prob=input_probs,
            next_prob=next_token_scores,
            **kwargs
        )
        return match_index, next_tokens, next_token_scores
    
    # Conditionally execute based on whether prefix_token_sampler is provided
    accept_index, next_tokens, next_probs = lax.cond(
        prefix_token_sampler is None,
        lambda: default_path(),
        lambda: sampler_path(input_tokens, next_tokens, input_probs, next_probs),
    )
    
    # Split tokens and scores based on matched_num
    acceptance_length = accept_index + 1
    matched_next_tokens = next_tokens[:, :acceptance_length]
    unmatched_next_tokens = next_tokens[:, acceptance_length:]
    matched_next_scores = next_probs[:, :acceptance_length]
    unmatched_next_scores = next_probs[:, acceptance_length:]
    
    return (
        acceptance_length,
        matched_next_tokens,
        unmatched_next_tokens,
        matched_next_scores,
        unmatched_next_scores
    )

# For adapt