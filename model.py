import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

_MODEL     = None
_TOKENIZER = None

_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

_SYSTEM_MESSAGE = (
    "You are an expert ML Research Assistant. "
    "You help users understand AI papers, model benchmarks, code, and the latest ML news. "
    "Always reply in English. Be concise and accurate. "
    "Never add hashtags or fabricate information."
)


def _load_model() -> None:
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return

    _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_ID)
    if _TOKENIZER.pad_token is None:
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    _MODEL = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    _MODEL.eval()
    print(f"Model loaded: {_MODEL_ID}")


def _generate(prompt: str) -> str:
    """Core generation logic — runs on GPU."""
    _load_model()

    messages = [
        {"role": "system", "content": _SYSTEM_MESSAGE},
        {"role": "user",   "content": prompt},
    ]
    input_text = _TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _TOKENIZER(input_text, return_tensors="pt")
    device = next(_MODEL.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = _MODEL.generate(
            **inputs,
            max_new_tokens=512,      # increased for detailed ML answers
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.3,
            eos_token_id=_TOKENIZER.eos_token_id,
        )

    input_len  = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][input_len:]
    return _TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


# Apply @spaces.GPU only when running on HuggingFace Spaces
# Falls back to plain function call when running locally / on Colab
if HAS_SPACES:
    generate_response = spaces.GPU(_generate)
else:
    generate_response = _generate
