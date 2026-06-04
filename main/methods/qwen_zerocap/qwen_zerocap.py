# methods/qwen_zerocap/qwen_zerocap.py

from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import open_clip
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from qwen_vl_utils import process_vision_info


DEFAULT_CAPTION_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def _freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def _to_legacy_past(past):
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    return tuple(past)


def _to_model_past(past):
    if hasattr(past, "get_seq_length"):
        return past

    # Some transformers versions have DynamicCache.from_legacy_cache, others do not.
    two_tensor_past = tuple(
        (layer[0], layer[1])
        for layer in past
    )

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(two_tensor_past)

    cache = DynamicCache()
    for layer_idx, (key_states, value_states) in enumerate(two_tensor_past):
        cache.update(key_states, value_states, layer_idx)

    return cache


def _add_past(past, delta):
    out = []
    for layer_past, layer_delta in zip(past, delta):
        layer_out = []
        for p, d in zip(layer_past, layer_delta):
            if p is None:
                layer_out.append(None)
            elif d is None:
                layer_out.append(p)
            else:
                layer_out.append(p + d)
        out.append(tuple(layer_out))
    return tuple(out)


def _zeros_like_past(past):
    out = []
    for layer in past:
        layer_out = []
        for x in layer:
            if x is None:
                layer_out.append(None)
            else:
                layer_out.append(torch.zeros_like(x, requires_grad=False))
        out.append(tuple(layer_out))
    return tuple(out)


def _requires_grad_past(delta):
    out = []
    for layer in delta:
        layer_out = []
        for x in layer:
            if x is None:
                layer_out.append(None)
            else:
                layer_out.append(x.detach().clone().requires_grad_(True))
        out.append(tuple(layer_out))
    return tuple(out)


def _detach_past(past):
    out = []
    for layer in past:
        layer_out = []
        for x in layer:
            if x is None:
                layer_out.append(None)
            else:
                layer_out.append(x.detach())
        out.append(tuple(layer_out))
    return tuple(out)


@dataclass
class QwenZeroCapConfig:
    caption_model: str = DEFAULT_CAPTION_MODEL
    device_map: str = "auto"
    dtype: str = "auto"
    attn_implementation: Optional[str] = None

    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_device: str = "cuda:0"

    prompt: str = "Write a caption for this image."
    clip_text_prefix: str = "Image of "

    max_new_tokens: int = 48
    min_new_tokens: int = 4
    top_size: int = 128
    num_iterations: int = 3

    clip_loss_temperature: float = 0.01
    clip_scale: float = 1.0
    ce_scale: float = 0.2
    stepsize: float = 0.3
    grad_norm_factor: float = 0.9
    fusion_factor: float = 0.99

    repetition_penalty: float = 1.1
    end_factor: float = 1.01
    verbose_cache: bool = True


class QwenZeroCapGenerator:
    def __init__(self, config: QwenZeroCapConfig):
        self.config = config

        qwen_kwargs = {
            "dtype": config.dtype,
            "device_map": config.device_map,
        }
        if config.attn_implementation is not None:
            qwen_kwargs["attn_implementation"] = config.attn_implementation

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.caption_model,
            **qwen_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(config.caption_model)
        self.tokenizer = self.processor.tokenizer

        _freeze(self.model)

        self.clip, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            config.clip_model_name,
            pretrained=config.clip_pretrained,
            device=config.clip_device,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(config.clip_model_name)
        _freeze(self.clip)

        self._cache_printed = False

        self.eos_token_ids = self._get_eos_token_ids()
        self.end_token_id = self._get_end_token_id(".")

    @property
    def input_device(self):
        return next(self.model.parameters()).device

    def _get_eos_token_ids(self):
        eos = self.tokenizer.eos_token_id
        if eos is None:
            return set()
        if isinstance(eos, list):
            return set(eos)
        return {int(eos)}

    def _get_end_token_id(self, text: str):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return None
        return int(ids[-1])

    def _build_inputs(self, image: Image.Image, prompt: str):
        image = image.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.input_device)

        return inputs

    def _replace_input_ids(self, base_inputs, input_ids):
        seq_len = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        base_seq_len = base_inputs["input_ids"].shape[1]

        inputs = {}

        for k, v in base_inputs.items():
            if k in ["input_ids", "attention_mask"]:
                continue

            # token_type_ids 같은 sequence-aligned tensor 처리
            if (
                torch.is_tensor(v)
                and v.ndim >= 2
                and v.shape[0] == batch_size
                and v.shape[1] == base_seq_len
            ):
                v = v.to(input_ids.device)

                if seq_len <= base_seq_len:
                    inputs[k] = v[:, :seq_len]
                else:
                    pad_shape = list(v.shape)
                    pad_shape[1] = seq_len - base_seq_len
                    pad = torch.zeros(
                        pad_shape,
                        dtype=v.dtype,
                        device=input_ids.device,
                    )
                    inputs[k] = torch.cat([v, pad], dim=1)
            else:
                inputs[k] = v

        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = torch.ones(
            (batch_size, seq_len),
            dtype=torch.long,
            device=input_ids.device,
        )

        return inputs

    def _print_cache_structure_once(self, past):
        if self._cache_printed or not self.config.verbose_cache:
            return

        past = _to_legacy_past(past)
        print("past type:", type(past))
        print("past len:", len(past))
        print("layer0 type:", type(past[0]))
        print("layer0 len:", len(past[0]))
        print("k shape:", past[0][0].shape)
        print("v shape:", past[0][1].shape)
        self._cache_printed = True

    @torch.no_grad()
    def _encode_clip_image(self, image: Image.Image):
        image_input = self.clip_preprocess(image.convert("RGB")).unsqueeze(0)
        image_input = image_input.to(self.config.clip_device)

        image_features = self.clip.encode_image(image_input)
        image_features = F.normalize(image_features, dim=-1)
        return image_features.detach()

    @torch.no_grad()
    def _encode_clip_texts(self, texts: List[str]):
        text_input = self.clip_tokenizer(texts).to(self.config.clip_device)
        text_features = self.clip.encode_text(text_input)
        text_features = F.normalize(text_features, dim=-1)
        return text_features.detach()

    def _decode_caption_ids(self, caption_ids: List[int]) -> str:
        if len(caption_ids) == 0:
            return ""
        return self.tokenizer.decode(
            caption_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _make_candidate_texts(self, caption_ids: List[int], candidate_ids: torch.Tensor):
        texts = []
        for token_id in candidate_ids.tolist():
            cand_ids = caption_ids + [int(token_id)]
            cand_text = self._decode_caption_ids(cand_ids)
            texts.append(self.config.clip_text_prefix + cand_text)
        return texts

    def _clip_target_loss(
        self,
        probs: torch.Tensor,
        caption_ids: List[int],
        image_features: torch.Tensor,
    ):
        top_size = min(self.config.top_size, probs.shape[-1])
        top_probs, top_indices = probs.topk(top_size, dim=-1)

        top_probs = top_probs.squeeze(0).clamp_min(1e-12)
        top_indices = top_indices.squeeze(0)

        candidate_texts = self._make_candidate_texts(caption_ids, top_indices)
        text_features = self._encode_clip_texts(candidate_texts)

        with torch.no_grad():
            image_features = image_features.to(text_features.device)
            sims = image_features @ text_features.T
            target_probs = F.softmax(
                sims.squeeze(0) / self.config.clip_loss_temperature,
                dim=-1,
            ).to(top_probs.device)

        loss = -(target_probs * torch.log(top_probs)).sum()
        return loss

    def _fluency_loss(self, probs: torch.Tensor, probs_before_shift: torch.Tensor):
        probs = probs.clamp_min(1e-12)
        probs_before_shift = probs_before_shift.clamp_min(1e-12)
        return (probs * (probs.log() - probs_before_shift.log())).sum(-1).sum()

    def _forward_last_token_with_past(
        self,
        last_token: torch.Tensor,
        past,
        total_context_len: int,
    ):
        # For Qwen3-VL cached decoding, past length is already encoded in DynamicCache.
        # Passing full-context attention_mask double-counts the prefix length.
        attention_mask = torch.ones(
            (last_token.shape[0], last_token.shape[1]),
            dtype=torch.long,
            device=last_token.device,
        )

        return self.model(
            input_ids=last_token,
            attention_mask=attention_mask,
            past_key_values=_to_model_past(past),
            use_cache=True,
            return_dict=True,
        )

    def _shift_past(
        self,
        past,
        last_token: torch.Tensor,
        total_context_len: int,
        probs_before_shift: torch.Tensor,
        caption_ids: List[int],
        image_features: torch.Tensor,
    ):
        past = _to_legacy_past(past)
        past = _detach_past(past)

        context_delta = _zeros_like_past(past)

        for _ in range(self.config.num_iterations):
            curr_delta = _requires_grad_past(context_delta)
            shifted_past = _add_past(past, curr_delta)

            with torch.enable_grad():
                outputs = self._forward_last_token_with_past(
                    last_token=last_token,
                    past=shifted_past,
                    total_context_len=total_context_len,
                )
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

                clip_loss = self._clip_target_loss(
                    probs=probs,
                    caption_ids=caption_ids,
                    image_features=image_features,
                )
                ce_loss = self._fluency_loss(probs, probs_before_shift)

                loss = (
                    self.config.clip_scale * clip_loss
                    + self.config.ce_scale * ce_loss
                )

                loss.backward()

            new_delta = []
            for delta_layer, grad_layer in zip(context_delta, curr_delta):
                new_layer = []
                for old_delta, grad_tensor in zip(delta_layer, grad_layer):
                    if old_delta is None or grad_tensor is None:
                        new_layer.append(None)
                        continue

                    grad = grad_tensor.grad
                    if grad is None:
                        new_layer.append(old_delta.detach())
                        continue

                    norm = torch.norm(grad) + 1e-15
                    update = (
                        -self.config.stepsize
                        * grad
                        / (norm ** self.config.grad_norm_factor)
                    )
                    new_layer.append((old_delta + update.detach()).detach())
                new_delta.append(tuple(new_layer))

            context_delta = tuple(new_delta)

        shifted_past = _add_past(past, context_delta)
        return _detach_past(shifted_past)

    def _apply_logit_penalties(
        self,
        logits: torch.Tensor,
        generated_ids: List[int],
        step: int,
    ):
        logits = logits.clone()

        for token_id in generated_ids[-4:]:
            token_id = int(token_id)
            if logits[0, token_id] > 0:
                logits[0, token_id] /= self.config.repetition_penalty
            else:
                logits[0, token_id] *= self.config.repetition_penalty

        if self.end_token_id is not None:
            if step < self.config.min_new_tokens:
                logits[0, self.end_token_id] -= 5.0
            else:
                logits[0, self.end_token_id] *= self.config.end_factor

        return logits

    def _should_stop(self, next_token_id: int, step: int):
        if step + 1 < self.config.min_new_tokens:
            return False

        if next_token_id in self.eos_token_ids:
            return True

        if self.end_token_id is not None and next_token_id == self.end_token_id:
            return True

        return False

    def _get_step_state(self, base_inputs, context_ids):
        with torch.no_grad():
            full_inputs = self._replace_input_ids(base_inputs, context_ids)
            full_outputs = self.model(
                **full_inputs,
                use_cache=True,
                return_dict=True,
            )

            self._print_cache_structure_once(full_outputs.past_key_values)

            logits_before_shift = full_outputs.logits[:, -1, :]
            probs_before_shift = F.softmax(logits_before_shift, dim=-1).detach()

            prefix_ids = context_ids[:, :-1]
            last_token = context_ids[:, -1:]

            prefix_inputs = self._replace_input_ids(base_inputs, prefix_ids)
            prefix_outputs = self.model(
                **prefix_inputs,
                use_cache=True,
                return_dict=True,
            )

            prefix_past = _to_legacy_past(prefix_outputs.past_key_values)

        return probs_before_shift, prefix_past, last_token

    def generate(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        if prompt is None:
            prompt = self.config.prompt
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens

        image = image.convert("RGB")
        image_features = self._encode_clip_image(image)
        base_inputs = self._build_inputs(image, prompt)
        base_input_ids = base_inputs["input_ids"]

        generated_ids: List[int] = []

        for step in range(max_new_tokens):
            if len(generated_ids) == 0:
                context_ids = base_input_ids
            else:
                gen = torch.tensor(
                    [generated_ids],
                    dtype=torch.long,
                    device=base_input_ids.device,
                )
                context_ids = torch.cat([base_input_ids, gen], dim=1)

            probs_before_shift, prefix_past, last_token = self._get_step_state(
                base_inputs=base_inputs,
                context_ids=context_ids,
            )

            shifted_past = self._shift_past(
                past=prefix_past,
                last_token=last_token,
                total_context_len=context_ids.shape[1],
                probs_before_shift=probs_before_shift,
                caption_ids=generated_ids,
                image_features=image_features,
            )

            with torch.no_grad():
                shifted_outputs = self._forward_last_token_with_past(
                    last_token=last_token,
                    past=shifted_past,
                    total_context_len=context_ids.shape[1],
                )
                shifted_logits = shifted_outputs.logits[:, -1, :]
                shifted_logits = self._apply_logit_penalties(
                    shifted_logits,
                    generated_ids=generated_ids,
                    step=step,
                )

                shifted_probs = F.softmax(shifted_logits, dim=-1)
                shifted_probs = shifted_probs.clamp_min(1e-12)
                probs_before = probs_before_shift.clamp_min(1e-12)

                final_probs = (
                    shifted_probs ** self.config.fusion_factor
                    * probs_before ** (1.0 - self.config.fusion_factor)
                )
                final_probs = final_probs / final_probs.sum(dim=-1, keepdim=True)

                next_token_id = int(final_probs.argmax(dim=-1).item())

            generated_ids.append(next_token_id)

            if self._should_stop(next_token_id, step):
                break

        caption = self._decode_caption_ids(generated_ids)
        return caption


if __name__ == "__main__":
    image = Image.new("RGB", (512, 512), color="white")

    config = QwenZeroCapConfig(
        max_new_tokens=16,
        top_size=64,
        num_iterations=1,
        clip_device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    generator = QwenZeroCapGenerator(config)
    caption = generator.generate(image)
    print(caption)