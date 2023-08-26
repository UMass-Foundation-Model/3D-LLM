## --------------------------------------------------------PART:Caption code---------------------------------------------------------- ##
import os
import re
import random
import glob
import yaml
from tqdm import tqdm
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    wait_incrementing,
)  # for exponential backoff
import openai
from PIL import Image
from enum import auto, Enum
from conversation import Conversation
from blip2 import Blip2
import torch
import clip


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    DOLLY = auto()
    OASST_PYTHIA = auto()
    BAIZE = auto()


QUESTION_INSTRUCTION = (
    "I have an image. "
    "Ask me questions about the content of this image. "
    "Carefully asking me informative questions to maximize your information about this image content. "
    "Each time ask one question only without giving an answer. "
    "Avoid asking yes/no questions."
    'I\'ll put my answer beginning with "Answer:".'
)
SUB_QUESTION_INSTRUCTION = "Next Question. Avoid asking yes/no questions. \n" "Question: "

SUMMARY_INSTRUCTION = (
    "Now summarize the information you get in a few sentences.\n "
    "Don't add the not sure or negative information into summary. \n"
    "Remember you do not need to summary all sentences. Ignore the sentences with answers negative or not sure.\n"
    "Don't imagine or add information.\n "
    "Don't add negative statement or not sure statement into summary.\n"
    "Summary: "
)

ANSWER_INSTRUCTION = "Answer given questions. If you are not sure about the answer, say you don't know honestly. Don't imagine any contents that are not in the image."

SUB_ANSWER_INSTRUCTION = "Answer: "  # template following blip2 huggingface demo

FIRST_QUESTION = "Descripe the 3D object in the photo."

DEFAULT_INSTRUCTION = (
    "A chat between a curious user and an artificial intelligence assistant. "
    + "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

VALID_CHATGPT_MODELS = ["gpt-3.5-turbo"]
VALID_GPT3_MODELS = ["text-davinci-003", "text-davinci-002", "davinci"]


def get_chat_log(questions, answers, last_n=-1):
    n_addition_q = len(questions) - len(answers)
    assert (n_addition_q) in [0, 1]
    template = "Question: {} \nAnswer: {} \n"
    chat_log = ""
    if last_n > 0:
        answers = answers[-last_n:]
        questions = questions[-(last_n + n_addition_q) :]
    elif last_n == 0:
        answers = []
        questions = questions[-1:] if n_addition_q else []

    for i in range(len(answers)):
        chat_log = chat_log + template.format(questions[i], answers[i])
    if n_addition_q:
        chat_log = chat_log + "Question: {}".format(questions[-1])
    else:
        chat_log = chat_log[:-2]  # remove the last '/n'
    return chat_log


def prepare_gpt_prompt(task_prompt, questions, answers, sub_prompt):
    gpt_prompt = "\n".join([task_prompt, get_chat_log(questions, answers), sub_prompt])
    return gpt_prompt


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_gpt3(
    gpt3_prompt, max_tokens=40, model="text-davinci-003"
):  # 'text-curie-001' does work at all to ask questions
    response = openai.Completion.create(model=model, prompt=gpt3_prompt, max_tokens=max_tokens)  # temperature=0.6,
    reply = response["choices"][0]["text"]
    total_tokens = response["usage"]["total_tokens"]
    return reply, total_tokens


def prepare_vicuna_message(conversation, task_prompt, questions, answers, sub_prompt):
    conversation.system = task_prompt
    conversation.messages = []
    conversation.offset = 0
    assert len(questions) == len(answers)
    for q, a in zip(questions, answers):  # message为一个列表，内容为：开始的问题，问题，回答，问题，回答。。。，下一个问题的指导
        conversation.messages.append(["ASSISTANT", q])
        conversation.messages.append(["USER", a])
    conversation.messages.append(["USER", sub_prompt])
    conversation.messages.append(["ASSISTANT", None])

    return conversation


def summarize_chat(
    questions, answers, model, max_gpt_token=100, vicuna_model=None, vicuna_tokenizer=None, conversation=None
):
    if model in VALID_GPT3_MODELS:
        summary_prompt = prepare_gpt_prompt(QUESTION_INSTRUCTION, questions, answers, SUMMARY_INSTRUCTION)
        summary, n_tokens = call_gpt3(summary_prompt, model=model, max_tokens=max_gpt_token)
    elif model in VALID_CHATGPT_MODELS:
        summary_prompt = prepare_chatgpt_message(QUESTION_INSTRUCTION, questions, answers, SUMMARY_INSTRUCTION)

        summary, n_tokens = call_chatgpt(summary_prompt, model=model, max_tokens=max_gpt_token)

    # 添加vicuna answer
    elif model == "vicuna":
        summary_prompt = prepare_vicuna_message(
            conversation, QUESTION_INSTRUCTION, questions, answers, SUMMARY_INSTRUCTION
        )

        prompt = conversation.get_prompt()

        inputs = vicuna_tokenizer([prompt])
        output_ids = vicuna_model.generate(
            torch.as_tensor(inputs.input_ids).cuda(),
            do_sample=True,
            temperature=0.7,
            max_new_tokens=1024,
        )
        outputs = vicuna_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
        summary = outputs[skip_echo_len:].strip()

        n_tokens = 0

    elif isinstance(model, Blip2):
        summary_prompt = prepare_gpt_prompt(QUESTION_INSTRUCTION, questions, answers, SUMMARY_INSTRUCTION)
        n_tokens = 0  # local model. no token cost on OpenAI API.
        summary = model.call_llm(summary_prompt)
    else:
        raise ValueError("{} is not a valid question model".format(model))

    summary = summary.replace("\n", " ").strip()
    return summary, summary_prompt, n_tokens


def prepare_chatgpt_message(task_prompt, questions, answers, sub_prompt):
    messages = [{"role": "system", "content": task_prompt}]

    assert len(questions) == len(answers)
    for q, a in zip(questions, answers):  # message为一个列表，内容为：开始的问题，问题，回答，问题，回答。。。，下一个问题的指导
        messages.append({"role": "assistant", "content": "Question: {}".format(q)})
        messages.append({"role": "user", "content": "Answer: {}".format(a)})
    messages.append({"role": "system", "content": sub_prompt})

    return messages


@retry(wait=wait_incrementing(start=1, increment=3), stop=stop_after_attempt(10))
def call_chatgpt(chatgpt_messages, max_tokens=40, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model, messages=chatgpt_messages, temperature=0.6, max_tokens=max_tokens
    )
    reply = response["choices"][0]["message"]["content"]
    total_tokens = response["usage"]["total_tokens"]
    return reply, total_tokens


class AskQuestions:
    def __init__(
        self,
        img,
        blip2,
        model,
        max_gpt_token=30,
        n_blip2_context=-1,
        vicuna_model=None,
        vicuna_tokenizer=None,
        conversation=None,
    ):
        self.img = img
        self.blip2 = blip2
        self.model = model
        self.max_gpt_token = max_gpt_token
        self.n_blip2_context = n_blip2_context

        self.questions = []
        self.answers = []
        self.total_tokens = 0
        self.vicuna_model = vicuna_model
        self.vicuna_tokenizer = vicuna_tokenizer
        self.conversation = conversation

    def reset(self, img):
        self.img = img
        self.questions = []
        self.answers = []
        self.total_tokens = 0

    def ask_question(self):
        if len(self.questions) == 0:
            # first question is given by human to request a general discription
            question = FIRST_QUESTION  # describe the image in details
        else:
            if self.model in VALID_CHATGPT_MODELS:
                chatgpt_messages = prepare_chatgpt_message(
                    QUESTION_INSTRUCTION, self.questions, self.answers, SUB_QUESTION_INSTRUCTION
                )
                question, n_tokens = call_chatgpt(chatgpt_messages, model=self.model, max_tokens=self.max_gpt_token)
            elif self.model in VALID_GPT3_MODELS:
                # prepare the context for GPT3
                gpt3_prompt = prepare_gpt_prompt(
                    QUESTION_INSTRUCTION, self.questions, self.answers, SUB_QUESTION_INSTRUCTION
                )

                question, n_tokens = call_gpt3(gpt3_prompt, model=self.model, max_tokens=self.max_gpt_token)

            elif self.model == "vicuna":
                prepare_vicuna_message(
                    self.conversation, QUESTION_INSTRUCTION, self.questions, self.answers, SUB_QUESTION_INSTRUCTION
                )
                prompt = self.conversation.get_prompt()

                inputs = self.vicuna_tokenizer([prompt])
                output_ids = self.vicuna_model.generate(
                    torch.as_tensor(inputs.input_ids).cuda(),
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                )
                outputs = self.vicuna_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
                question = outputs[skip_echo_len:].strip()

                n_tokens = 0

            elif isinstance(self.model, Blip2):
                # prepare the context for other LLM
                gpt_prompt = prepare_gpt_prompt(
                    QUESTION_INSTRUCTION, self.questions, self.answers, SUB_QUESTION_INSTRUCTION
                )
                n_tokens = 0  # local model. no token cost on OpenAI API.
                question = self.model.call_llm(gpt_prompt)
            else:
                raise ValueError("{} is not a valid question model".format(self.model))

            self.total_tokens = self.total_tokens + n_tokens

        return question

    def question_trim(self, question):
        question = question.split("Question: ")[-1].replace("\n", " ").strip()
        if "Answer:" in question:  # Some models make up an answer after asking. remove it
            q, a = question.split("Answer:")[:2]
            if len(q) == 0:  # some not so clever models will put the question after 'Answer:'.
                question = a.strip()
            else:
                question = q.strip()
        return question

    def answer_question(self):
        # prepare the context for blip2
        blip2_prompt = "\n".join(
            [
                ANSWER_INSTRUCTION,
                get_chat_log(self.questions, self.answers, last_n=self.n_blip2_context),
                SUB_ANSWER_INSTRUCTION,
            ]
        )
        # print("blip2_prompt", blip2_prompt)
        answer = self.blip2.ask(self.img, blip2_prompt)
        return answer

    def answer_trim(self, answer):
        answer = answer.split("Question:")[0].replace("\n", " ").strip()
        return answer

    def chatting(self, n_rounds, print_mode):
        if print_mode == "chat":
            print("--------Chat Starts----------")

        for i in tqdm(range(n_rounds), desc="Chat Rounds", disable=print_mode != "bar"):
            question = self.ask_question()

            question = self.question_trim(question)
            self.questions.append(question)

            if print_mode == "chat":
                if self.model == "vicuna":
                    print("vicuna: {}".format(question))
                else:
                    print("GPT-3: {}".format(question))

            elif print_mode == "gradio":
                gr_chatbot = gr_chatbot + [[question, None]]

            answer = self.answer_question()
            answer = self.answer_trim(answer)
            self.answers.append(answer)

            if print_mode == "chat":
                print("BLIP-2: {}".format(answer))
            elif print_mode == "gradio":
                self.gr_chatbot[-1][1] = answer

        if print_mode == "chat":
            print("--------Chat Ends----------")

        return self.questions, self.answers, self.total_tokens


def caption_image(
    blip2,
    image,
    model,
    args,
    n_rounds=10,
    n_blip2_context=-1,
    print_mode="no",
    vicuna_model=None,
    vicuna_tokenizer=None,
):
    conv_vicuna_v1_1 = None
    if model == "gpt3":
        model = "text-davinci-003"
    elif model == "chatgpt":
        model = "gpt-3.5-turbo"

    if model == "vicuna":
        conv_vicuna_v1_1 = Conversation(
            system=QUESTION_INSTRUCTION,
            roles=["USER", "ASSISTANT"],
            messages=[],
            offset=0,
            sep_style=SeparatorStyle.TWO,
            sep=" ",
            sep2="</s>",
        ).copy()

    results = {}

    if model == "vicuna":
        chat = AskQuestions(
            image,
            blip2,
            n_blip2_context=n_blip2_context,
            model=model,
            vicuna_model=vicuna_model,
            vicuna_tokenizer=vicuna_tokenizer,
            conversation=conv_vicuna_v1_1,
        )
    else:
        chat = AskQuestions(image, blip2, n_blip2_context=n_blip2_context, model=model)

    questions, answers, n_token_chat = chat.chatting(n_rounds, print_mode=print_mode)

    if n_rounds > 1:
        if model == "vicuna":
            summary, summary_prompt, n_token_sum = summarize_chat(
                questions,
                answers,
                model=model,
                conversation=conv_vicuna_v1_1,
                vicuna_tokenizer=vicuna_tokenizer,
                vicuna_model=vicuna_model,
            )
        else:
            summary, summary_prompt, n_token_sum = summarize_chat(questions, answers, model=model)

        if model == "vicuna":
            results["vicuna Captioner"] = {
                "caption": summary,
                "chat": summary_prompt,
                "n_token": n_token_chat + n_token_sum,
            }
        else:
            results["ChatGPT Captioner"] = {
                "caption": summary,
                "chat": summary_prompt,
                "n_token": n_token_chat + n_token_sum,
            }
        if print_mode != "no":
            print("---------- Multi-round chatting for captioning one image ----------")
            print(f"Prompt for summarizing one image:\n{summary_prompt}\n\n")
            print(f"Summary of one image:\n{summary}")
            print("-------------------------------------------------------------------")
    else:
        summary = answers[0]

    results["BLIP2+OurPrompt"] = {"caption": answers[0]}

    # Default BLIP2 caption
    caption = blip2.caption(image)
    results["BLIP2"] = {"caption": caption}

    return results, summary


def caption_images_from_dir(
    blip2s,
    model,
    image_dir,
    args,
    save_path="",
    n_rounds=10,
    n_blip2_context=-1,
    print_mode="no",
    vicuna_model=None,
    vicuna_tokenizer=None,
    clip_select=False,
    views=["outside"],
):
    """
    Caption images with a set of blip2 models

    Args:
        blip2s (dict): A dict of blip2 models. Key is the blip2 model name
        model (str or Blip2): the model name used to ask quetion. Valid values are 'gpt3', 'chatgpt', and their concrete model names
                    including 'text-davinci-003', 'davinci,' and 'gpt-3.5-turbo'.
                    If passing a Blip2 instance, will use its backend LLM.
        save_path (str): the path to save caption results. If it is empty, results are not being saved.
        n_rounds (int): the number of chat rounds
        n_blip2_context (int): how many previous QA rounds can blip2 see. negative value means blip2 can see all
        print_mode (str): print mode. 'chat' for printing everying. 'bar' for printing everthing but the chat process. 'no' for no printing
    """
    if clip_select:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

    if model == "gpt3":
        model = "text-davinci-003"
    elif model == "chatgpt":
        model = "gpt-3.5-turbo"

    caption_dict = {}
    img_caption = {}
    for view in views:
        image_list = glob.glob(f"{image_dir}/rgb_{view}*.png")
        image_list = [os.path.basename(path) for path in image_list]
        image_list = sorted(image_list, key=lambda i: int(re.findall(r"\d+", i)[0]))
        image_list = random.sample(image_list, 4)
        image_list = sorted(image_list, key=lambda i: int(re.findall(r"\d+", i)[0]))
        print("image list", image_list)

        description_list = []
        for image_name in image_list:
            img_path = image_dir + "/" + image_name
            image = Image.open(img_path).convert("RGB")
            info = {"setting": {"dataset": image_dir, "n_rounds": n_rounds}}

            if clip_select:
                view_caption_list = []
                for _ in range(2):
                    for blip2_tag, blip2 in blip2s.items():
                        info[blip2_tag], summary = caption_image(
                            blip2,
                            image,
                            args=args,
                            n_rounds=n_rounds,
                            n_blip2_context=n_blip2_context,
                            model=model,
                            print_mode=print_mode,
                            vicuna_model=vicuna_model,
                            vicuna_tokenizer=vicuna_tokenizer,
                        )
                    view_caption_list.append(summary)
                # do view caption clip selection here
                description_tokens = clip.tokenize(view_caption_list).to(device)
                description_image = clip_preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits_per_image, logits_per_text = clip_model(description_image, description_tokens)
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

                    max_index = probs.tolist().index(max(probs))
                print(f"logic before softmax: {logits_per_image}\nlogic after softmax: {probs}")
                summary = view_caption_list[max_index]
            else:
                for blip2_tag, blip2 in blip2s.items():
                    info[blip2_tag], summary = caption_image(
                        blip2,
                        image,
                        args=args,
                        n_rounds=n_rounds,
                        n_blip2_context=n_blip2_context,
                        model=model,
                        print_mode=print_mode,
                        vicuna_model=vicuna_model,
                        vicuna_tokenizer=vicuna_tokenizer,
                    )

            description_list.append(summary)

            img_caption.update({image_name[:-4]: summary})

            if print_mode != "no":
                pass

        description_prompt = f"{description_list[0]} {description_list[1]} {description_list[2]} {description_list[3]}"
        system_prompt_overall = f"You are a 3D object descriptor. Given four different descriptions of the same object from different viewpoints, summarize a concrete description in several sentences. Avoid uncertain or negative information. Avoid describing the background. \n\n\
\
The four descriptions are as follows.  {description_prompt}\n\n\
\
Concrete 3D object description:"

        if model == "vicuna":
            conv_vicuna_summary = Conversation(
                system=system_prompt_overall,
                roles=["USER", "ASSISTANT"],
                messages=[],
                offset=0,
                sep_style=SeparatorStyle.TWO,
                sep=" ",
                sep2="</s>",
            ).copy()

            prompt = conv_vicuna_summary.get_prompt()

            inputs = vicuna_tokenizer([prompt])
            output_ids = vicuna_model.generate(
                torch.as_tensor(inputs.input_ids).cuda(),
                do_sample=True,
                temperature=0.7,
                max_new_tokens=1024,
            )
            outputs = vicuna_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
            outputs = outputs[skip_echo_len:].strip()
            if print_mode != "no":
                print("####### Summarizing multiple images for captioning this scene #######")
                print(f"Prompt for summarizing this scene:\n{system_prompt_overall}")
                print(f"Summary of this scene:\n{outputs}")
                print("#####################################################################")

            if view == "inside":
                caption_dict.update({"scene_caption_inside": outputs})
            else:
                caption_dict.update({"scene_caption_outside": outputs})

        elif model in VALID_GPT3_MODELS:
            summary_prompt = prepare_gpt_prompt(
                system_prompt_overall, [description_prompt], [None], SUMMARY_INSTRUCTION
            )

            summary, n_tokens = call_gpt3(summary_prompt, model=model, max_tokens=1024)
            if print_mode != "no":
                print(f"{summary}")
            caption_dict.update({"scene_caption_outside": summary})

        elif model in VALID_CHATGPT_MODELS:
            summary_prompt = prepare_chatgpt_message(
                system_prompt_overall, [description_prompt], [None], SUMMARY_INSTRUCTION
            )

            summary, n_tokens = call_chatgpt(summary_prompt, model=model, max_tokens=1024)

            if print_mode != "no":
                print(f"{summary}")
            caption_dict.update({"scene_caption_outside": summary})

    caption_dict.update({"img_caption": img_caption})
    return caption_dict


## --------------------------------------------------------PART:Caption code---------------------------------------------------------- ##
