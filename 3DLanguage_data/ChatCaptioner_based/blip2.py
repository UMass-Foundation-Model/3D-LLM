## --------------------------------------------------------PART:BLIP2 model code---------------------------------------------------------- ##
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor
import torch


BLIP2DICT = {
    "FlanT5 XXL": "Salesforce/blip2-flan-t5-xxl",
    "FlanT5 XL COCO": "Salesforce/blip2-flan-t5-xl-coco",
    "OPT6.7B COCO": "Salesforce/blip2-opt-6.7b-coco",
    "OPT2.7B COCO": "Salesforce/blip2-opt-2.7b-coco",
    "FlanT5 XL": "Salesforce/blip2-flan-t5-xl",
    "OPT6.7B": "Salesforce/blip2-opt-6.7b",
    "OPT2.7B": "Salesforce/blip2-opt-2.7b",
}


class Blip2:
    def __init__(self, model, device_id, bit8=True):
        # load BLIP-2 to a single gpu
        self.tag = model
        self.bit8 = bit8
        self.device = "cuda:{}".format(device_id)

        dtype = {"load_in_8bit": True} if self.bit8 else {"torch_dtype": torch.float16}
        self.blip2_processor = AutoProcessor.from_pretrained(BLIP2DICT[self.tag])
        self.blip2 = Blip2ForConditionalGeneration.from_pretrained(
            BLIP2DICT[self.tag], device_map={"": device_id}, **dtype
        )
        # self.blip2 = Blip2ForConditionalGeneration.from_pretrained(BLIP2DICT[self.tag], device_map='auto', load_in_8bit=True)

    def ask(self, raw_image, question):
        inputs = self.blip2_processor(raw_image, question, return_tensors="pt").to(self.device, torch.float16)
        out = self.blip2.generate(**inputs)
        answer = self.blip2_processor.decode(out[0], skip_special_tokens=True)
        return answer

    def nucleus_ask(self, raw_image, question):
        inputs = self.blip2_processor(raw_image, question, return_tensors="pt").to(self.device, torch.float16)
        # print(self.blip2)
        print(inputs)
        generated_ids = self.blip2.generate(
            **inputs,
            # pixel_values=inputs.pixel_values,
            do_sample="Nucleus sampling",
            temperature=0.5,
            length_penalty=-1.0,
            repetition_penalty=1.0,
            max_length=30,
            min_length=1,
            num_beams=5,
            top_p=0.1
        )
        print(generated_ids)
        result = self.blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return result

    def caption(self, raw_image, mode="default"):
        if mode == "default":
            # starndard way to caption an image in the blip2 paper
            caption = self.ask(raw_image, "a photo of")
            caption = caption.replace("\n", " ").strip()  # trim caption
        elif mode == "nucleus":
            caption = self.nucleus_ask(raw_image, "a photo of")
        return caption

    def call_llm(self, prompts):
        prompts_temp = self.blip2_processor(None, prompts, return_tensors="pt")
        input_ids = prompts_temp["input_ids"].to(self.device)
        attention_mask = prompts_temp["attention_mask"].to(self.device, torch.float16)

        prompts_embeds = self.blip2.language_model.get_input_embeddings()(input_ids)

        outputs = self.blip2.language_model.generate(inputs_embeds=prompts_embeds, attention_mask=attention_mask)

        outputs = self.blip2_processor.decode(outputs[0], skip_special_tokens=True)

        return outputs


## --------------------------------------------------------PART:BLIP2 model code---------------------------------------------------------- ##
