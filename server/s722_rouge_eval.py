import os
import sys
from rouge import Rouge

# 设置项目的root路径，方便项目文件的导入
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

my_path = root_path + '/src'
print('my_path--->', my_path)
sys.path.append(my_path) # 需要使用绝对路径  # 注意sys.path.append("./src") # 相对路径不起作用

from server.s721_predict import Predict
from src.d14_func_utils import timer
import src.d10_config as d10_config

print('root_path--->', root_path)


# 构建 ROUGE 评估类代码
class RougeEval():
    def __init__(self, path):
        self.path = path
        self.scores = None
        self.rouge = Rouge()
        self.sources = []   # 原文source，对soruce进行预测生成摘要，存放在 hypos列表 中
        self.hypos = []     # 假定摘要 - 待生成的机器摘要存储空间，也就是机器摘要
        self.refs = []      # 人工摘要 - 从文件中读取人工摘要
        self.process()      # 预处理

    # 从文件中读取 机器摘要 人工摘要，并测试样本个数
    def process(self):
        print('Reading from ', self.path)
        with open(self.path, 'r') as test:
            for line in test:
                source, ref = line.strip().split('<SEP>')
                ref = ref.replace('。', '.')
                self.sources.append(source)
                self.refs.append(ref)

        print('self.refs[]包含的样本数: ', len(self.refs))
        print(f'Test set contains {len(self.sources)} samples.')


    # 生成预测集合：根据source生成机器摘要，存放在hypos列表中
    @timer('building hypotheses')
    def build_hypos(self, predict):
        # Generate hypos for the dataset.
        print('Building hypotheses.')
        count = 0
        for source in self.sources:
            count += 1
            if count % 1000 == 0:
                print('count=', count)

            # 调用模型输入原始文本，产生摘要
            myres = predict.predict(source.split())
            # print(myres)
            self.hypos.append(myres)


    # 获取平均分数的函数
    def get_average(self):
        assert len(self.hypos) > 0, '需要首先构建hypotheses。Build hypotheses first!'
        print('Calculating average rouge scores.')
        # 输入机器摘要 、人工摘要 求rouge平均分数
        return self.rouge.get_scores(self.hypos, self.refs, avg=True)

    def one_sample(self, hypo, ref):
        return self.rouge.get_scores(hypo, ref)[0]


if __name__ == '__main__':

    # 真实的测试机是val_data_path: dev.txt(3000条)
    print('实例化Rouge对象 ... ')
    rouge_eval = RougeEval(d10_config.val_data_path100)

    print('实例化Predict对象 ... ')
    predict = Predict()

    # 利用模型对article进行预测
    print('利用模型对article进行预测, 并通过Rouge对象进行评估 ... ')
    rouge_eval.build_hypos(predict)

    # 将预测结果和标签abstract进行ROUGE规则计算
    print('开始用Rouge规则进行评估 ... ')
    result = rouge_eval.get_average()
    print('result-->', result)
    print('rouge1: ', result['rouge-1'])
    print('rouge2: ', result['rouge-2'])
    print('rougeL: ', result['rouge-l'])

    # 最后将计算评估结果写入文件中
    print('将评估结果写入结果文件中 ... ')
    with open('../eval_result/rouge_result.txt', 'a') as f:
        for r, metrics in result.items():
            f.write(r + '\n')
            for metric, value in metrics.items():
                f.write(metric + ': ' + str(value * 100) + '\n')



    ''' 
    result--> 
    {'rouge-1': {'r': 0.32206730396814853, 'p': 0.4905012716042128, 'f': 0.35217523352899294}, 
    'rouge-2': {'r': 0.1110961628388486, 'p': 0.1269196184515743, 'f': 0.10333588939821019}, 
    'rouge-l': {'r': 0.29625100501288243, 'p': 0.4466587350557939, 'f': 0.3226774485736824}}
    '''
