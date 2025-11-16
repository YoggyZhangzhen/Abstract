import os
import sys
import torch
import time

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

my_path = root_path + '/src'
print('my_path--->', my_path)
sys.path.append(my_path) # 需要使用绝对路径  # 注意sys.path.append("./src") # 相对路径不起作用

import  src.d10_config as config
from server.s721_predict import Predict
from server.s722_rouge_eval  import RougeEval

if __name__ == '__main__':

    # 真实的测试机是val_data_path: dev.txt
    print('实例化Rouge对象......')
    rouge_eval = RougeEval(config.val_data_path100)
    print('实例化Predict对象......')
    predict = Predict()

    # 利用模型对article进行预测
    print('利用模型对article进行预测, 并通过Rouge对象进行评估......')
    rouge_eval.build_hypos(predict)

    # 将预测结果和标签abstract进行ROUGE规则计算
    print('开始用Rouge规则进行评估......')
    result = rouge_eval.get_average()

    print('rouge1: ', result['rouge-1'])
    print('rouge2: ', result['rouge-2'])
    print('rougeL: ', result['rouge-l'])

    # 最后将计算评估结果写入文件中
    print('将评估结果写入结果文件中......')
    with open(root_path + '/w/rouge_result_greedy_quantized_cpu.txt', 'w') as f:
        for r, metrics in result.items():
            f.write(r + '\n')
            for metric, value in metrics.items():
                f.write(metric + ': ' + str(value * 100) + '\n')


    print('预测推理 End')
