from absl.app import run
from tqdm import tqdm
import imageio
import numpy as np
from PIL import Image
from transformers import GenerationConfig, AutoTokenizer
import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from tux import (
    define_flags_with_default, StreamingCheckpointer, JaxDistributedConfig,
    set_random_seed, get_float_dtype_by_name, JaxRNG,
    match_partition_rules, make_shard_and_gather_fns,
    with_sharding_constraint, tree_apply, next_rng
)
from lwm.vision_llama import VideoLLaMAConfig, FlaxVideoLLaMAForCausalLM
from lwm.vqgan import VQGAN
from lwm.sjd import debug,get_random_100_pairs,get_first_100_pairs
import time
import pickle
import os

FLAGS, FLAGS_DEF = define_flags_with_default(
    prompt='MSRVTT',#'VBench' 'MSRVTT'
    output_file='',
    temperature_image=1.0,
    temperature_video=1.0,
    top_k_image=8192,
    top_k_video=100,
    cfg_scale_image=1.0,
    cfg_scale_video=1.0,
    vqgan_checkpoint='',
    n_frames=1,
    seed=1234,
    mesh_dim='1,-1,1,1',
    dtype='fp32',
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    tokenizer='LargeWorldModel/LWM-Text-1M',
    llama=VideoLLaMAConfig.get_default_config(),
    jax_distributed=JaxDistributedConfig.get_default_config(),
    # TODO: Update parameters of SJD
    rand_token_num=32,
    prefix_token_sampler_scheme='speculative_jacobi',
    prefill_way="fsjd",
    benchmark_path="/data/lei/dataset/MSRVTT", 
    eval_len=2,
    benchmark_way="order"
)
def calculate_mean_nonzero(combined_array):
    # 提取非零元素
    non_zero_elements = combined_array[combined_array != 0]
    
    # 计算非零元素的均值
    mean_non_zero = np.mean(non_zero_elements)
    
    return mean_non_zero

def main(argv):
    assert FLAGS.output_file != ''

    JaxDistributedConfig.initialize(FLAGS.jax_distributed)
    set_random_seed(FLAGS.seed)

    tokens_per_frame = 257
    vqgan = VQGAN(FLAGS.vqgan_checkpoint, replicate=False)
    mesh = VideoLLaMAConfig.get_jax_mesh(FLAGS.mesh_dim)
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer, local_files_only=True, legacy=False)
    prefix_tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer, truncation_side='left', padding_side='left', legacy=False)
    tokenizer.pad_token_id = 0
    prefix_tokenizer.pad_token_id = 0
    if FLAGS.load_llama_config != '':
        llama_config = VideoLLaMAConfig.load_config(FLAGS.load_llama_config)
        updates = VideoLLaMAConfig(**FLAGS.llama)
        llama_config.update(dict(
            scan_attention=updates.scan_attention,
            scan_mlp=updates.scan_mlp,
            scan_query_chunk_size=updates.scan_query_chunk_size,
            scan_key_chunk_size=updates.scan_key_chunk_size,
            scan_mlp_chunk_size=updates.scan_mlp_chunk_size,
            scan_layers=updates.scan_layers,
            param_scan_axis=updates.param_scan_axis,
        ))
    else:
        llama_config = VideoLLaMAConfig(**FLAGS.llama)

    if FLAGS.update_llama_config != '':
        llama_config.update(dict(eval(FLAGS.update_llama_config)))

    llama_config.update(dict(
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    ))
    llama_config.update(dict(mesh_dim=FLAGS.mesh_dim))

    # NOTE: SP的超参数
    llama_config.update(dict(
        rand_token_num=FLAGS.rand_token_num,
        prefill_way=FLAGS.prefill_way,
        prefix_token_sampler_scheme=FLAGS.prefix_token_sampler_scheme,
    ))
    with jax.default_device(jax.devices("cpu")[0]):
        _, params = StreamingCheckpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, disallow_trainstate=True, max_buffer_size=32 * 2 ** 30
        )
        # llama_config, params = debug(llama_config, params, layer=2, scan_layers=True)
        #NOTE:fix a bug input_shape=(512, 8192)-->input_shape=(256, llama_config.max_sequence_length)
        model = FlaxVideoLLaMAForCausalLM(
            llama_config,
            input_shape=(4, llama_config.max_sequence_length),#(512, 8192),
            seed=FLAGS.seed,
            _do_init=False,
            dtype=get_float_dtype_by_name(FLAGS.dtype),
        )
        model_ps = match_partition_rules(
            VideoLLaMAConfig.get_partition_rules(llama_config.scan_layers, llama_config.param_scan_axis), params
        )
        shard_fns, _ = make_shard_and_gather_fns(
            model_ps, get_float_dtype_by_name(FLAGS.dtype)
        )

        with mesh:
            params = tree_apply(shard_fns, params)

    def _forward_generate(params, rng, batch, n_tokens, cfg_scale, top_k, temperature):
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp'), 'sp'))
        cfg_scales = jnp.ones((batch['input_ids'].shape[0] // 2,), dtype=jnp.float32) * cfg_scale
        cfg_scales = with_sharding_constraint(cfg_scales, PS(('dp', 'fsdp')))
        rng_generator = JaxRNG(rng)
        output = model.generate_vision(
            batch['input_ids'],
            cfg_scales,
            attention_mask=batch['attention_mask'],
            vision_masks=batch['vision_masks'],
            params=params['params'],
            prng_key=rng_generator(),
            generation_config=GenerationConfig(
                max_new_tokens=n_tokens,
                min_new_tokens=n_tokens,
                pad_token_id=tokenizer.pad_token_id,
                temperature=temperature,
                do_sample=True,
                top_k=top_k,
            )
        ).sequences[:, batch['input_ids'].shape[1]:]
        return output, rng_generator()
    _sharded_forward_generate = pjit(
        _forward_generate,
        in_shardings=(model_ps, PS(), PS()),
        out_shardings=(PS(), PS()),
        static_argnums=(3, 4, 5, 6)
    )

    # Generate an image or first frame (for video)
    def generate_first_frame(prompts, max_input_length):
        nonlocal sharded_rng
        uncond_prompts = ["<s><vision>"] * len(prompts)
        prompts = prompts + uncond_prompts
        inputs = prefix_tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_input_length,
            return_tensors='np'
        )
        batch = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            vision_masks=np.zeros(inputs.input_ids.shape, dtype=bool),
        )
        with mesh:
            output, sharded_rng = _sharded_forward_generate(
                params, sharded_rng, batch,
                tokens_per_frame, FLAGS.cfg_scale_image,
                FLAGS.top_k_image, FLAGS.temperature_image
            )
            output_bio = jax.device_get(output)
            output = np.split(output_bio, 2, axis=0)[0]
            acceptance_length_list = np.split(output_bio, 2, axis=0)[1]
        output = output.reshape(len(prompts) // 2, tokens_per_frame)
        image = vqgan.decode(output[:, :-1].reshape(-1, 16, 16))
        image = ((jax.device_get(image) + 1) * 127.5).astype(np.uint8)
        return output, image, acceptance_length_list

    sharded_rng = next_rng()
    video_name = []
    # ++++++++++++++++++++++++++Benchmark_init++++++++++++++++++++++++++
    if FLAGS.prompt == 'MSRVTT':
        import csv
        data,prompts = [],[]
        csvfile = f"{FLAGS.benchmark_path}/MSRVTT_JSFUSION_test.csv"
        with open(csvfile, "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                data.append(row)
        if FLAGS.benchmark_way == 'random':
            data = get_random_100_pairs(data[1:],len=FLAGS.eval_len)
        else:
            data = get_first_100_pairs(data[1:],len=FLAGS.eval_len)
        for i, row in enumerate(data):
            prompts.append(row[-1])
            video_name.append(row[-2])
    else:
        prompts = [FLAGS.prompt]
    entries = []
    for prompt in prompts:
        entries.append({
            'caption': prompt,
            'prompt': f"<s>You are a helpful assistant. USER: Generate an image of {prompt} ASSISTANT: <vision>",
        })

    B = 1
    first_acceptance_length_list, later_acceptance_length_list = [],[]
    time_list_first, time_list_later = [],[]
    images, image_encodings = [], []
    print("Begin generate First image")
    for i in tqdm(list(range(0, len(entries), B))):
        entries_i = entries[i:i + B]
        prompts = [entry['prompt'] for entry in entries_i]
        print(f"No.{i} is {prompts}")
        st = time.time()
        img_enc, img, acceptance_length_list = generate_first_frame(prompts, max_input_length=128)
        time_list_first.append(time.time() - st)
        first_acceptance_length_list.append(acceptance_length_list)
        image_encodings.extend(img_enc)
        images.extend(img)

    if FLAGS.n_frames == 1:
        image = images[0]
        Image.fromarray(image).save(FLAGS.output_file)
        return

    # Generate the rest of the video
    def generate_video_pred(prompts, images, max_input_length):
        nonlocal sharded_rng
        images = np.concatenate([images, images], axis=0)
        uncond_prompts = ["<s><vision>"] * len(prompts)
        prompts = prompts + uncond_prompts
        inputs = prefix_tokenizer(
            prompts,
            padding='max_length',
            truncation=True,
            max_length=max_input_length,
            return_tensors='np'
        )
        batch = dict(
            input_ids=np.concatenate([inputs.input_ids, images], axis=1),
            attention_mask=np.concatenate([inputs.attention_mask, np.ones(images.shape, dtype=inputs.attention_mask.dtype)], axis=1),
            vision_masks=np.concatenate([
                np.zeros(inputs.input_ids.shape, dtype=bool),
                np.ones(images.shape, dtype=bool)
            ], axis=1),
        )
        with mesh:
            output, sharded_rng = _sharded_forward_generate(
                params, sharded_rng, batch,
                (FLAGS.n_frames - 1) * tokens_per_frame, FLAGS.cfg_scale_video,
                FLAGS.top_k_video, FLAGS.temperature_video
            )
            output_bio = jax.device_get(output)
            output = np.split(output_bio, 2, axis=0)[0]
            acceptance_length_list = np.split(output_bio, 2, axis=0)[1]
        output = output.reshape(len(prompts) // 2, FLAGS.n_frames - 1, tokens_per_frame)
        output = np.concatenate([images[:len(prompts) // 2, None], output], axis=1)
        output = output[:, :, :-1].reshape(-1, FLAGS.n_frames, 16, 16)
        vision = []
        for v in output:
            v = vqgan.decode(v)
            v = ((jax.device_get(v) + 1) * 127.5).astype(np.uint8)
            vision.append(v)
        return vision, acceptance_length_list

    new_entries = []
    for img_enc, entry in zip(image_encodings, entries):
        new_entries.append({
            'caption': entry['caption'],
            'prompt': f"<s>You are a helpful assistant. USER: Generate a video of {entry['caption']} ASSISTANT: <vision>",
            'image': np.array(img_enc, dtype=np.int32),
        })
    entries = new_entries

    B = 1
    videos = []
    print("Begin generate ALL Frame")
    for i in tqdm(list(range(0, len(entries), B))):
        entries_i = entries[i:i + B]
        prompts = [entry['prompt'] for entry in entries_i]
        images = np.array([entry['image'] for entry in entries_i], dtype=np.int32)
        st = time.time()
        video, acceptance_length_list = generate_video_pred(prompts, images, max_input_length=128)
        time_list_later.append(time.time() - st)
        later_acceptance_length_list.append(acceptance_length_list)
        videos.extend(video)

    # ++++++++++++++++++++++++++Save_result++++++++++++++++++++++++++
    import json
    all_acceptance_length_list = []
    file_out_path = '/home/leihaodong/AAAI25/exp/LWMSJD/test'
    file_out_path = FLAGS.output_file
    file_out_video_path = os.path.join(file_out_path, 'video')
    os.makedirs(file_out_path, exist_ok=True)
    os.makedirs(file_out_video_path, exist_ok=True)
    # Config save
    with open(os.path.join(file_out_path,"llama_config.json"), 'w') as f:
        json.dump(llama_config.to_dict(), f, indent=4)
        print("llama_config saves in ",os.path.join(file_out_path,"llama_config.json"))

    # Accl & Time save
    result_data = {}
    mean_time, mean_accl = 0,0
    for i in range(len(later_acceptance_length_list)):
        ACCL_list, time_i = [], 0
        prompt_key = f"prompt_{i}"
        ACCL_list.append(first_acceptance_length_list[i])
        ACCL_list.append(later_acceptance_length_list[i])
        
        time_i = time_list_first[i] + time_list_later[i]
        ACCL_list = np.concatenate(ACCL_list, axis=1)
        
        all_acceptance_length_list.append(ACCL_list)
        
        mean_accl_i = calculate_mean_nonzero(all_acceptance_length_list[i])
        new_entry = {
            "prompt": entries[i]['caption'],
            "times": time_i,  # 转换为 Python float
            "acc_l": mean_accl_i,       # 转换为 Python float
        }
        mean_time += time_i
        mean_accl += mean_accl_i
        print(f"all_time:{time_i:.2f} mean_accl:{mean_accl_i:.2f}")
        result_data[prompt_key] = new_entry
    
    mean_time, mean_accl = mean_time/len(later_acceptance_length_list), mean_accl/len(later_acceptance_length_list)
    new_result_data = {"Summerary": {"mean_time": mean_time, "mean_accl": mean_accl}}
    new_result_data.update(result_data)

    with open(os.path.join(file_out_path,"result.json"), 'w') as f:
        json.dump(new_result_data, f, indent=4)
    with open(os.path.join(file_out_path,"accl_list.pkl"), 'wb') as f:
        pickle.dump(all_acceptance_length_list, f)

    # Video save
    for i,video in enumerate(videos):
        if len(video_name)>=i:
            name = f"{video_name[i]}.mp4"
        else:
            name = f"{i}.mp4"
        writer = imageio.get_writer(os.path.join(file_out_video_path,name), fps=4)
        for frame in video:
            writer.append_data(frame)
        writer.close()

    print('done')

if __name__ == "__main__":
    run(main)
