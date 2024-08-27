# -*- coding: utf-8 -*-
"""
usage:
CUDA_VISIBLE_DEVICES=0 python fastapi_server_demo.py --model_type bloom --base_model bigscience/bloom-560m

curl -X 'POST' 'http://0.0.0.0:8008/chat' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "咏鹅--骆宾王；登黄鹤楼--"
}'
"""
import json
import argparse
import os
from threading import Thread

from matplotlib.font_manager import json_dump
import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger
from peft import PeftModel
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
)
from fastapi.responses import StreamingResponse
from template import get_conv_template
from utils import setup_log
logger = setup_log("ailog")

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}


# logger.add("logs/api_logs.log", level="INFO")  # 所有级别为INFO及以上的日志都会被记录到文件


@torch.inference_mode()
def stream_generate_answer_label(
        model,
        tokenizer,
        prompt,
        device,
        do_print=True,
        max_new_tokens=512,
        repetition_penalty=1.0,
        context_len=2048,
        stop_str="</s>",
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        stop = False
        pos = new_text.find(stop_str)
        if pos != -1:
            new_text = new_text[:pos]
            stop = True
        generated_text += new_text
        if do_print:
            print(new_text, end="", flush=True)
        if stop:
            break
    if do_print:
        print()
    return generated_text

@torch.inference_mode()
def stream_generate_answer(
        model,
        tokenizer,
        prompt,
        device,
        do_print=True,
        max_new_tokens=512,
        repetition_penalty=1.0,
        context_len=2048,
        stop_str="</s>",
):
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    input_ids = tokenizer(prompt).input_ids
    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    generation_kwargs = dict(
        input_ids=torch.as_tensor([input_ids]).to(device),
        max_new_tokens=max_new_tokens,
        num_beams=1,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        stop = False
        pos = new_text.find(stop_str)
        if pos != -1:
            new_text = new_text[:pos]
            stop = True
        #generated_text += new_text
        yield new_text
        if do_print:
            print(new_text, end="", flush=True)
        if stop:
            break
    if do_print:
        print()
    #return generated_text


class Item(BaseModel):
    input: str = Field(..., max_length=2048)
    history: list = Field(..., allow_null=True)  # 添加history属性



def main():
    system_prompt_label = '''
# 角色
你是一位高效的问题分类专家，专门负责将用户提出的问题迅速而准确地归类为两大领域：健康相关与非健康相关。

## 技能
### 技能1：关键词与主题识别
- **快速解析**：接收用户输入的问题后，立即识别其中的关键信息和主题。
- **精准定位**：利用内置的关键词库和智能算法，准确定位问题的核心领域。

### 技能2：类别判断与输出
- **即时分类**：根据识别出的关键词和主题，判断问题属于“健康相关”或“非健康相关”类别。
- **简洁反馈**：仅输出对应类别的标签，无需额外解释或建议。

## 工作流程
1. **接收输入**：获取用户提交的问题。
2. **分析内容**：通过关键词匹配和语境理解，识别问题焦点。
3. **类别判定**：基于分析结果，决定问题的分类。
4. **输出结果**：直接返回类别标签：“健康相关”或“非健康相关”。

## 限制
- 分类决策严格基于问题内容，不涉及对用户背景的假设或额外信息的查询。
- 仅提供问题的宽泛分类，不涉及深入解答或建议。
- 忽略问题中可能存在的模糊性或双重含义，以最直接关联的类别为准。

## 示例互动
- **用户问题**：我应该如何开始我的早晨冥想练习？
  - **回答**：健康相关

- **用户问题**：推荐一本关于人工智能的入门书籍。
  - **回答**：非健康相关

- **用户问题**：最近睡眠质量很差，有没有改善建议？
  - **回答**：健康相关

- **用户问题**：明天的天气怎么样？
  - **回答**：非健康相关
'''
    system_prompt = '''- Role: 专业的营养师，名为Leafye AI营养师
- Background: 用户正在寻找个性化的营养建议，可能受到健康问题的影响。
- Profile: Leafye AI营养师是一位精通个性化营养建议的人工智能专家，致力于根据用户的具体需求提供专业的指导和建议。
- Skills: 拥有广泛的营养学知识和能力，能够提供针对性的营养建议和健康问题分析。
- Goals: 旨在为用户提供针对其特定健康问题的营养建议，协助用户改善健康状况。
- Constrains: 建议必须基于科学研究和实践经验，避免传播未经验证的信息。
- OutputFormat: 文本回复格式。
- Workflow:
  1. 接收并分析用户的健康问题描述。
  2. 根据问题分析提供专业的营养建议。
- Examples:
  用户问题："我最近经常感到疲劳，有什么营养建议吗？"
  建议回复："根据您的情况，建议您增加富含铁和维生素B群的食物摄入，如瘦肉、绿叶蔬菜和全谷物。这些营养素有助于提高能量水平。"
  
  用户问题："北京奥运会在哪一年？"
  建议回复："北京奥运会是在2008年举办的。如果你对奥运会的历史、运动项目或健康与运动相关的话题感兴趣，随时可以问我！"

  用户问题："我正在尝试减肥，需要哪些营养调整？"
  建议回复："减肥时，建议关注整体饮食平衡，增加蛋白质摄入并减少加工食品和高糖食品的摄入。同时，保持适量运动也是关键。"
- Initialization: 欢迎使用Leafye AI营养师服务，我是您的个性化营养顾问。请告诉我您的健康问题，我将为您提供专业的营养建议。'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str, required=True)
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument('--template_name', default="vicuna", type=str,
                        help="Prompt template name, eg: alpaca, vicuna, baichuan, chatglm2 etc.")
    parser.add_argument('--system_prompt', default=system_prompt_label, type=str)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--max_new_tokens", default=512, type=int)
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--gpus', default="0", type=str)
    parser.add_argument('--only_cpu', action='store_true', help='only use CPU for inference')
    parser.add_argument('--port', default=8009, type=int)
    args = parser.parse_args()
    print(args)

    def load_model(args):
        if args.only_cpu is True:
            args.gpus = ""
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        load_type = 'auto'
        if torch.cuda.is_available():
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
            print("Loaded lora model")
        else:
            model = base_model
        if device == torch.device('cpu'):
            model.float()
        model.eval()
        print(tokenizer)
        return model, tokenizer, device

    # define the app
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    model, tokenizer, device = load_model(args)
    prompt_template = get_conv_template(args.template_name)
    stop_str = tokenizer.eos_token if tokenizer.eos_token else prompt_template.stop_str
    def predict(history, sentence):
        # history = [[sentence, '']]

        def get_label():
            label_prompt = prompt_template.get_prompt(messages=[[sentence, ""]], system_prompt=system_prompt_label)
            label_response = stream_generate_answer_label(
                model,
                tokenizer,
                label_prompt,
                device,
                do_print=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                stop_str=stop_str,
            )

            return label_response.strip()
        label = get_label()
        if label not in ["健康相关", "非健康相关"]:
            label = "非健康相关"
        history_messages = history + [[sentence, ""]]
        prompt = prompt_template.get_prompt(messages=history_messages, system_prompt=system_prompt)
        response_iterator = stream_generate_answer(
            model,
            tokenizer,
            prompt,
            device,
            do_print=False,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=args.repetition_penalty,
            stop_str=stop_str,
        )
            # 格式化为SSE消息
        response_str = ""
        for response in response_iterator:
            # 创建包含 label 和 response 的字典
            data_dict = {'label': label, 'response': response}
            response_str = response_str + response
            # 序列化字典为 JSON 字符串
            json_data = json.dumps(data_dict, ensure_ascii=False)
            yield f"data: {json_data}\n\n"

        # 然后，将拼接后的字符串记录到日志中
        logger.info(f"问题：{sentence}")
        logger.info(f"回复：{label}: {response_str}")

    @app.get('/')
    async def index():
        return {"message": "index, docs url: /docs"}

    @app.post('/chat')
    async def chat(item: Item):
        try:
            response_iterator = predict(item.history, item.input)
            #result_dict = {'response': response}
            # logger.info(f"Successfully get result, input:{item.input}")
            # logger.info(f"response:{response}")
            return response_iterator
        
        except Exception as e:
            logger.error(e)
            return None
  
    @app.get('/chat_get')
    async def chat_get(history: str = None, sentence: str = None):
        if history is None or sentence is None:
            return {"error": "Missing query parameters"}
        try:
            # 解析查询参数中的 history JSON 字符串
            history_list = json.loads(history)
            # 你的 predict 函数实现
            response_iterator = predict(history_list, sentence)
            # 创建一个 StreamingResponse，设置为 EventStream
            response = StreamingResponse(
                response_iterator,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                status_code=200
            )
            response.flush_after_write = True  # 确保每次写入后都会发送数据
            return response
        except Exception as e:
            logger.error(e)
            # 返回500错误响应
            return StreamingResponse(
                ["Internal Server Error"],
                media_type="text/plain",
                status=500
            )
        
    @app.post('/chat_post')
    async def chat_post(item: Item):
        try:
            response_iterator = predict(item.history, item.input)


            # 创建一个 StreamingResponse，设置为 EventStream
            response = StreamingResponse(
                response_iterator,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                status_code=200
            )
            response.flush_after_write = True  # 确保每次写入后都会发送数据
            return response
        except Exception as e:
            logger.error(e)
            # 返回500错误响应
            return StreamingResponse(
                ["Internal Server Error"],
                media_type="text/plain",
                status=500
            )
    uvicorn.run(app=app, host='0.0.0.0', port=args.port, workers=1)


if __name__ == '__main__':
    main()
