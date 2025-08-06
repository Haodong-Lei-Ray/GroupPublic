from absl.app import run
import numpy as np
from tux import (
    define_flags_with_default, JaxDistributedConfig,
    set_random_seed,
)
from lwm.vision_llama import VideoLLaMAConfig
from lwm.vqgan import VQGAN
from lwm.sjd import debug,get_random_pairs,get_order_pairs
import jax
import jax.numpy as jnp
import os

FLAGS, FLAGS_DEF = define_flags_with_default(
    vqgan_checkpoint='',
    seed=1234,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='/data/lei/localmodel/LargeWorldModel/LWM-Chat-1M-Jax/vqgan',
    tokenizer='LargeWorldModel/LWM-Text-1M',
    llama=VideoLLaMAConfig.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
    save_path='/home/leihaodong/AAAI25/ckpt/LWM-Chat-1M-Jax'
)
def calculate_mean_nonzero(combined_array):
    # 提取非零元素
    non_zero_elements = combined_array[combined_array != 0]
    
    # 计算非零元素的均值
    mean_non_zero = np.mean(non_zero_elements)
    
    return mean_non_zero

def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    vqgan = VQGAN(FLAGS.vqgan_checkpoint, replicate=False)
    all_z = jnp.arange(vqgan.config.num_embeddings)
    # input_shape = (1, 256, 256, 3)  # 假设输入张量的形状为 (batch_size, height, width, channels)
    # variables = vqgan.model.init(jax.random.PRNGKey(0), jnp.ones(input_shape, dtype=jnp.float32))
    # vqgan.model.apply({'params': vqgan.params}, pixel_values, method=vqgan.model.encode)

    # 编码输入张量
    latents = vqgan.return_codebook(all_z)

    # 计算距离矩阵
    distances = jnp.linalg.norm(latents[:, None, :] - latents[None, :, :], axis=-1)  # (8192, 8192)
    # 将对角线元素设置为无穷大
    distances = distances.at[jnp.diag_indices_from(distances)].set(jnp.inf)

    # k-nearest neighbors
    k = latents.shape[0] - 1
    topk_indices = jax.lax.top_k(-distances, k=k)[1]  # 使用负值来获取最小值
    topk_indices_uint16 = topk_indices.astype(jnp.uint16)

    # 保存结果
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    np.save(os.path.join(FLAGS.save_path, f"top_{k}_indices.npy"), topk_indices_uint16)

if __name__ == "__main__":
    run(main)
