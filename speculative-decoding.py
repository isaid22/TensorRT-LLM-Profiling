### Generate Text Using Lookahead Decoding 
### Reference: https://nvidia.github.io/TensorRT-LLM/llm-api-examples/llm_lookahead_decoding.html#generate-text-using-lookahead-decoding
from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (LLM, BuildConfig, KvCacheConfig,
                                 LookaheadDecodingConfig, SamplingParams)

from nvtx import annotate

@annotate("TinyLlama-1p1B-inference", color="blue")
def main():

    # The end user can customize the build configuration with the build_config class
    build_config = BuildConfig()
    build_config.max_batch_size = 32

    checkpoint = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # checkpoint = "EleutherAI/pythia-1.4b-deduped"

    # The configuration for lookahead decoding
    lookahead_config = LookaheadDecodingConfig(max_window_size=4,
                                               max_ngram_size=4,
                                               max_verification_set_size=4)

    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)
    llm = LLM(model=checkpoint, dtype="bfloat16",
              kv_cache_config=kv_cache_config,
              build_config=build_config,
              speculative_config=lookahead_config,
              enable_build_cache=True)
    
    # Access TensorRT engine and check binding dtype
    # if hasattr(llm, 'engine'):  # Check if the LLM API exposes the TensorRT engine
    #     engine = llm._ngine
    #     for idx in range(engine.num_bindings):
    #         name = engine.get_binding_name(idx)
    #         dtype = engine.get_binding_dtype(idx)
    #         shape = engine.get_binding_shape(idx)
    #         is_input = engine.binding_is_input(idx)
    #         print(f"Binding {idx}: Name={name}, Dtype={dtype}, Shape={shape}, IsInput={is_input}")
    # else:
    #     print("LLM engine is not directly accessible.")
    print(dir(llm))  # List all attributes and methods of the LLM object

    prompt = "NVIDIA is a great company because"
    print(f"Prompt: {prompt!r}")

    sampling_params = SamplingParams(lookahead_config=lookahead_config)

    output = llm.generate(prompt, sampling_params=sampling_params)
    print(output)

    # Save engine to local disk
    # llm.save(checkpoint)

    #Output should be similar to:
    # Prompt: 'NVIDIA is a great company because'
    #RequestOutput(request_id=2, prompt='NVIDIA is a great company because', prompt_token_ids=[1, 405, 13044, 10764, 338, 263, 2107, 5001, 1363], outputs=[CompletionOutput(index=0, text='they are always pushing the envelope. They are always trying to make the best graphics cards and the best processors. They are always trying to make the best', token_ids=[896, 526, 2337, 27556, 278, 427, 21367, 29889, 2688, 526, 2337, 1811, 304, 1207, 278, 1900, 18533, 15889, 322, 278, 1900, 1889, 943, 29889, 2688, 526, 2337, 1811, 304, 1207, 278, 1900], cumulative_logprob=None, logprobs=[], finish_reason='length', stop_reason=None, generation_logits=None)], finished=True)
    # Check the precision of your TensorRT engine (if accessible)



if __name__ == '__main__':
    main()