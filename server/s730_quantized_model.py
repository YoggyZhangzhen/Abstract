import sys
import os
import torch
import time

'''
1 模型量化概念：在机器学习（深度学习）领域，模型量化一般是指将模型参数由类型FP32转换为INT8的过程，
    转换之后的模型大小会减少，所需内存、所需计算量会减少
2 pytorch  torch.quantization.quantize_dynamic()函数，对模型中的某些层进行量化
3 观察检查量化后各层参数
4 量化后的模型，只能用于推理验证，不能训练!!! 执行梯度迭代操作时会报错
'''
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
print('root_path--->', root_path)

my_path = root_path + '/src'
print('my_path--->', my_path)

sys.path.append(my_path) # 需要使用绝对路径
# sys.path.append("./src") # 相对路径不起作用


import src.d10_config as d10_config
from src.m13_model import PGN
import src.m11_vocab as vocab
from src.m12_dataset import PairDataset
from src.d14_func_utils import timer


# 打印模型大小
def print_size_of_model(model):
    # 保存模型中的参数部分到持久化文件
    torch.save(model.state_dict(), "temp.p")
    # 打印持久化文件的大小
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    # 移除该文件
    os.remove('temp.p')


# @timer(module='initalize and quantize model...')
def quantize_model(origin_model_path):
    # 第一步: 构造数据集字典
    dataset = PairDataset(d10_config.train_data_path,
                          max_enc_len=d10_config.max_enc_len,
                          max_dec_len=d10_config.max_dec_len,
                          truncate_enc=d10_config.truncate_enc,
                          truncate_dec=d10_config.truncate_dec)

    # 第二步: 利用字典, 实例化模型对象
    vocab = dataset.build_vocab(embed_file=d10_config.embed_file)
    model = PGN(vocab)

    # 判断待加载的模型是否存在
    if not os.path.exists(origin_model_path):
        print('The model file is not exists!')
        exit(0)

    # 将在GPU上训练好的模型加载到CPU上
    print('-------------------------------------------------------------------')
    model.load_state_dict(torch.load(origin_model_path, map_location=lambda storage, loc:storage))
    print('没有量化之前的模型 model--->', model)
    model.to('cpu')

    print('-------------------------------------------------------------------')
    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    # quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear, torch.nn.LSTM, torch.nn.Embedding}, dtype=torch.qint8)
    print('量化之后的模型 quantized_model--->', quantized_model)

    print('\n没有量化之前的模型大小', end='')
    print_size_of_model(model)

    print('\n量化之后的模型大小',  end='')
    print_size_of_model(quantized_model)

    # print('-------------------------------------------------------------------')
    # v1 = vars(quantized_model)
    # print('v1--->', v1)
    # v2 = vars(quantized_model.attention)
    # print('v2--->', v2)


'''
量化函数原型：Q = torch.quantize_per_tensor(input, scale = 0.025 , zero_point = 0, dtype = torch.quint8)
1 input为准备量化的32位浮点数，Q为量化后的8位定点数   （把32位的浮点数转成8位的数，节省存储空间 加快运算速度 降低精度）

2 dtype为量化类型，quint8代表8位无符号数；qint8代表8位带符号数，最高位是符号位
    二进制         0000,1000  代表十进制的8
    二进制         0000,1100  代表十进制的12
    quint8      【0，255】
    qint8 2^7   【-128，127】
    
3 假设量化为qint8, 设量化后的数Q为0001_1101,最高位为0（符号位），所以是正数；
	后7位转换为10进制是29，
	所以Q代表的数为 ：zero_point + Q * scale = 0 + 29 * 0.025 = 0.725

4 所以最终使用print显示Q时，显示的不是0001_1101而是0.725，但它在计算机中存储时，是0001_1101

5 使用dequantize()可以解除量化

6 量化公式：
	Q = round( (input-zero_point)/scale )
	29 = （x - 0）/ 0.025 ===> x = 0.725
	
	以zero_point为中心，用8位数Q代表input离中心有多远，scale为距离单位
    即input ≈ zero_point + Q * scale
    
7 所谓的 动态量化：有系统函数自动的指定 zero_point 和 scale。
'''
def dm01_test_torch_quantize_per_tensor():

    input = torch.rand(2, 3)
    print('input--->', input.shape, '\n', input)

    Q = torch.quantize_per_tensor(input, scale=0.025, zero_point=0, dtype=torch.quint8)

    # print("Q--->1", Q)
    print('Q--->2 量化后的数据类型', Q.dtype, '量化前的数据类型', Q.dequantize().dtype)
    print (Q.int_repr() )


if __name__ == '__main__':

    quantize_model(d10_config.model_save_path)
    # dm01_test_torch_quantize_per_tensor()
    print('模型量化 end')

