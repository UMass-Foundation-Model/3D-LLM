"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
from positional_encodings.torch_encodings import PositionalEncoding1D


@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.Qformer, self.query_tokens = self.init_Qformer(num_query_token, 1408)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)

        location_tokens = []
        for i in range(64):
            location_tokens.append("<loc%d>" % i)
        self.opt_tokenizer.add_special_tokens({"additional_special_tokens": location_tokens})

        self.opt_model = OPTForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer))
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]

        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, self.opt_model.config.hidden_size)

        self.max_txt_len = max_txt_len
        self.prompt = ""
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        pos_model = PositionalEncoding1D(1408 // 3)
        x = torch.zeros(1, 256, 1408 // 3)
        self.pos_embedding = pos_model(x).squeeze().cuda()

    def forward(self, samples):
        pc_embeds = samples["pc_feat"]

        pc = samples["pc"].long()
        all_pcs = torch.zeros((pc_embeds.shape))
        for j in range(pc.shape[0]):
            pcs = []
            for i in range(3):
                pc_i = pc[j][:, i]
                pcs.append(self.pos_embedding[pc_i])
            pcs = torch.cat(pcs, -1)
            all_pcs[j][:, :1407] = pcs
        all_pcs = all_pcs.cuda()

        pc_embeds = torch.cat([pc_embeds, all_pcs], 1)
        image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        self.opt_tokenizer.padding_side = "left"
        prompt = self.prompt
        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        text = [t for t in text_input]

        answers = [(t + a + "\n") for (t, a) in zip(text, samples["answer"])]

        opt_tokens = self.opt_tokenizer(
            answers,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(pc_embeds.device)

        idxes = (torch.where(opt_tokens.input_ids == 1948)[1] + 2).cpu().numpy().tolist()

        output_tokens = self.opt_tokenizer(
            answers,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(pc_embeds.device)

        targets = output_tokens.input_ids.masked_fill(output_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100)

        empty_targets = torch.ones(atts_opt.size(), dtype=torch.long).to(pc_embeds.device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        with self.maybe_autocast():
            pc_embeds = samples["pc_feat"]
            image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)

            query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=pc_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(pc_embeds.device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * pc_embeds.size(0)

            opt_tokens = self.opt_tokenizer(prompt, return_tensors="pt").to(pc_embeds.device)
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=False)
            output_text = [text.strip() for text in output_text]
            return output_text

    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        use_nucleus_sampling=False,
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=-1,
        repetition_penalty=1.0,
        **kwargs,
    ):
        with self.maybe_autocast():
            pc_embeds = samples["pc_feat"]

            pc = samples["pc"].long()
            all_pcs = torch.zeros((pc_embeds.shape))
            for j in range(pc.shape[0]):
                pcs = []
                for i in range(3):
                    pc_i = pc[j][:, i]
                    pcs.append(self.pos_embedding[pc_i])
                pcs = torch.cat(pcs, -1)
                all_pcs[j][:, :1407] = pcs
            all_pcs = all_pcs.cuda()
            pc_embeds = torch.cat([pc_embeds, all_pcs], 1)
        image_atts = torch.ones(pc_embeds.size()[:-1], dtype=torch.long).to(pc_embeds.device)
        query_tokens = self.query_tokens.expand(pc_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=pc_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(pc_embeds.device)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]
        if self.prompt:
            text_input = [self.prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        text = [t for t in text_input]

        input_tokens = self.opt_tokenizer(text, padding="longest", return_tensors="pt").to(pc_embeds.device)
        input_ids = input_tokens.input_ids
        encoder_atts = torch.cat([atts_opt, input_tokens.attention_mask], dim=1)

        if use_nucleus_sampling:
            input_embeds = inputs_opt.repeat_interleave(1, dim=0)
        # num_beams = 1
        else:
            input_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        with self.maybe_autocast(dtype=torch.float16):
            outputs = self.opt_model.generate(
                input_ids=input_ids,
                query_embeds=input_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
                # eos_token_id=self.eos_token_id,
            )

            prompt_length = input_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=False)
        return output_text

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            opt_model=opt_model,
            prompt=prompt,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model
