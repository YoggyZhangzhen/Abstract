# 导入必备工具包
import os
import sys
import torch
import time
import jieba

# 设定项目的root路径, 方便后续相关代码文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

my_path = root_path + '/src'
print('my_path--->', my_path)
sys.path.append(my_path) # 需要使用绝对路径  # 注意sys.path.append("./src") # 相对路径不起作用

# 导入项目相关的代码文件
import src.d10_config as d10_config

# 导入专门针对CPU的预测类Predict
from src.m23_predict import Predict

# 导入发送http请求的requests工具
import requests

# 服务框架使用Flask, 导入工具包
from flask import Flask
from flask import request
app = Flask(__name__)


# 加载自定义的停用词字典
jieba.load_userdict(d10_config.stop_word_file)

# 实例化Predict对象, 用于推断摘要, 提供服务请求
predict = Predict()
print('预测类Predict实例化完毕...')

# 设定 文本摘要服务的 路由和请求方法
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    # 接收来自请求方发送的服务字段
    uid = request.form['uid']
    text = request.form['text']

    # 对请求文本进行处理
    article = jieba.lcut(text)

    # 调用预测类对象执行摘要提取
    res = predict.predict(article)
    print('接受到一次请求 并处理', uid)

    return res


# 启动后台服务器
# 切换目录
    # eg: cd /Users/bombing/PycharmProjects/pythonProject3/text_summary/90_pgn_coverage_bm_server/server
# 启动服务
    # gunicorn -w 1 -b 0.0.0.0:5000 app:app
    # 其中-w设置最大进程数，-b绑定IP和端口，第一个app为app.py的文件名，第二个app为Flask应用的实例名。
