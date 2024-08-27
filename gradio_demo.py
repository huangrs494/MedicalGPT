# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

pip install gradio>=3.50.2
"""
import argparse
from threading import Thread

import gradio as gr
import torch
from peft import PeftModel
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    GenerationConfig,
    TextIteratorStreamer,
)

from template import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

import logging

# 设置日志配置
logging.basicConfig(filename='logs/chatbot_logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    system_prompt = "你扮演一位专业的营养师，名字为leafye AI营养师，专注于为用户提供个性化的营养补充建议。\
    你的交流方式模拟真实的营养咨询流程，根据用户的特定需求提供指导和建议。仅当用户询问与健康和营养相关的问题后，\
    在回复的结尾换行（另外新起一行），新增括号里面的内容，每个回复只新增一次（如需更详细的建议，点击以下链接进行了解#小程序://LEAFYE精准营养/首页/UwDTtUljdiWIkZi）\
    当用户的问题与健康和营养不相关时，回复内容不新增链接内容，\
    健康和营养相关的问题如：高血压、高血糖、高血脂、肌肉酸痛、耐力提升、骨骼健康、关节炎症、关节酸痛、冠心病、\
    中风、脑卒中、心血管健康、皮肤干燥、皱纹过多、夜间苏醒、免疫力地下、感冒、生病回复慢、消化不良、胀气、记忆力减退、\
    记忆力不集中、压力大、情绪不稳定、脱发、体力下降、精力不足、疲劳、腰酸、腿疼、眼睛酸痛、想减肥、想补充维生素、\
    睡眠不足、失眠、头疼、同型半胱氨酸、基因筛查等健康相关问题。"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='auto', type=str)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan2, chatglm2 etc.")
    parser.add_argument('--system_prompt', default=system_prompt, type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--share', action='store_true', help='Share gradio')
    parser.add_argument('--port', default=8081, type=int, help='Port of gradio demo')
    args = parser.parse_args()
    print(args)
    load_type = 'auto'
    if torch.cuda.is_available() and not args.only_cpu:
        device = torch.device(0)
    else:
        device = torch.device('cpu')

    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    base_model = model_class.from_pretrained(
        args.base_model,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        trust_remote_code=True,
    )
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.resize_emb:
        model_vocab_size = base_model.get_input_embeddings().weight.size(0)
        tokenzier_vocab_size = len(tokenizer)
        print(f"Vocab of the base model: {model_vocab_size}")
        print(f"Vocab of the tokenizer: {tokenzier_vocab_size}")
        if model_vocab_size != tokenzier_vocab_size:
            print("Resize model embeddings to fit tokenizer")
            base_model.resize_token_embeddings(tokenzier_vocab_size)
    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("loaded lora model")
    else:
        model = base_model
    if device == torch.device('cpu'):
        model.float()
    model.eval()
    prompt_template = get_conv_template(args.template_name)
    system_prompt = args.system_prompt
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str

    def predict(message, history):
        """Generate answer from prompt with GPT and stream the output"""
        history_messages = history + [[message, ""]]
        prompt = prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)
        streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = tokenizer(prompt).input_ids
        context_len = 2048
        max_new_tokens = 512
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(device),
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.0,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        partial_message = ""
        for new_token in streamer:
            if new_token != stop_str:
                partial_message += new_token
                yield partial_message
        logging.info(f"User: {message}")
        logging.info(f"AI: {partial_message}")

    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(),
        textbox=gr.Textbox(placeholder="Ask me question", lines=4, scale=9),
        title="leafye-AI营养师",
        description="我是一名专业的营养师，专注于为用户提供个性化的营养补充建议。欢迎交流咨询营养健康问题",
        theme="soft",
    ).queue().launch(share=args.share, inbrowser=True, server_name='0.0.0.0', server_port=args.port)


if __name__ == '__main__':
    main()
