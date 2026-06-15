import os

import sglang as sgl
from transformers import AutoTokenizer


def main():
    model_path = os.getenv("MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B")
    dllm_algorithm_config = os.getenv("DLLM_ALGORITHM_CONFIG")
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did Natalia sell altogether in April "
        "and May? Answer with the final number."
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )

    llm = sgl.Engine(
        model_path=model_path,
        trust_remote_code=True,
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config=dllm_algorithm_config,
        max_running_requests=1,
        context_length=1024,
        disable_cuda_graph=True,
        attention_backend="flashinfer",
    )
    try:
        output = llm.generate(
            prompt,
            {
                "temperature": 0.0,
                "max_new_tokens": 128,
            },
        )
        print(output)
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
