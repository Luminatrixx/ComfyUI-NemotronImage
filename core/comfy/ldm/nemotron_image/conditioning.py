from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import render_jinja_template


SYSTEM_PROMPT = (
    "Generate an image matching the requested resolution by detailing the color, "
    "quantity, text, shape, size, texture, spatial relationships of the objects and background."
)
TOKENIZER_KEY = "nemotron_tokenizer_json"
TEMPLATE_KEY = "nemotron_chat_template"


@dataclass(frozen=True)
class Prompt:
    text: str
    tokenizer: "ImageTokenizer"


class ImageTokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data=None):
        self.source = {key: tokenizer_data[key] for key in (TOKENIZER_KEY, TEMPLATE_KEY)}
        self.tokenizer = Tokenizer.from_str(self.source[TOKENIZER_KEY].numpy().tobytes().decode("utf-8"))
        self.template = self.source[TEMPLATE_KEY].numpy().tobytes().decode("utf-8")

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        return Prompt(text, self)

    def state_dict(self):
        return self.source.copy()

    def image_prompt_ids(self, text, width, height):
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate a {width}x{height} image.\n\n{text}"},
        ]
        rendered, _ = render_jinja_template(
            [conversation], chat_template=self.template, add_generation_prompt=True, enable_thinking=False,
            bos_token="<s>", eos_token="<|im_end|>", pad_token="<unk>", unk_token="<unk>",
        )
        return self.encode(rendered[0])


class TextConditioner(torch.nn.Module):
    dtypes = set()

    def __init__(self, dtype=None, device=None, model_options=None):
        super().__init__()
        self.register_buffer("conditioning", torch.zeros((1, 1, 1), dtype=torch.float32, device=device))

    def reset_clip_options(self):
        pass

    def set_clip_options(self, options):
        pass

    def encode_token_weights(self, prompt):
        return self.conditioning, None, {"nemotron_prompt": prompt}
