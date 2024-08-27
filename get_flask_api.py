import requests
import json
from urllib.parse import urlencode

# 定义查询参数
history = [
    [
        "北京奥运会哪一年",
        "非健康相关：北京奥运会是在2008年举办的。"
    ]
]
sentence = "最近总是生病，免疫力不好，怎么办"

# 将history列表转换为JSON字符串
history_json = json.dumps(history)

# 使用requests发送GET请求
response = requests.get(
    f"http://47.106.116.223:8009/chat_get",
    params={"history": history_json, "sentence": sentence}
)

# 打印响应内容
print(response.text)