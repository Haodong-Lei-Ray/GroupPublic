# for i in range(1, L):# draft token的位置0是上一个token, 所以一定是接受的
                # draft_token_index = draft_token_index_selector(i)
                # target_token_index = next_token_index_selector(i)
                # cls_idx = draft_tokens[b, draft_token_index]
                # # 1. 获得当前索引的概率
                # sampled_target_prob = next_prob[b, target_token_index, cls_idx]
                # sampled_draft_prob = draft_prob[b, draft_token_index, cls_idx]
                # r = rs[b, i, cls_idx]
                # def accept_fn(key_i):
                #     # 2.1 如果接受, last的拒绝的index不变
                #     rejected_index_list.at[b,i].set(rejected_idx)
                #     return (
                #         next_tokens_i.at[b, target_token_index].set(cls_idx),
                #         next_probs_i.at[b, target_token_index].set(draft_prob[b, draft_token_index]),
                #         rejected_idx,
                #         key_i,
                #     )

                # def reject_fn(key_i):
                #     # 2.1 如果接受, last的拒绝的index更新
                #     rejected_index_list.at[b,i].set(i)
                #     key_i, subkey = random.split(key_i)

                #     resampled_tokens, resampled_scores = self.reject_sampling_single_token(
                #         token_next_prob=next_prob[b, target_token_index],
                #         token_draft_prob=draft_prob[b, draft_token_index],
                #         logits_processor=logits_processor,
                #         logits_warper=logits_warper,
                #         all_collected_input_ids=jnp.concatenate([
                #             all_collected_input_ids[b],
                #             next_tokens_i[b, :target_token_index]
                #         ], axis=-1),
                #         key=subkey
                #     )

                #     return (#这里理论上不该改变target的分布的, 但是根据公式, 其要改变.
                #         next_tokens_i.at[b, target_token_index].set(resampled_tokens),
                #         next_probs_i.at[b, target_token_index].set(resampled_scores),
                #         i,
                #         key_i,
                #     )

                # # 2. 进行推测性采样
                # next_tokens_i, next_probs_i, rejected_idx, key_i = jax.lax.cond(
                #     r < jnp.minimum(sampled_target_prob / sampled_draft_prob, 1.0),
                #     accept_fn,
                #     reject_fn,
                #     key_i
                # )

            # return next_tokens_i, next_probs_i, rejected_index_list, key_i