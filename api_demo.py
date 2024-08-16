import argparse
import os
from threading import Thread

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

from template import get_conv_template

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, AutoTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

import configparser

# 创建配置解析器
config = configparser.ConfigParser()

# 读取配置文件
config.read('config.ini')

model_type = config['model_type']
model_type = config['base_model']

