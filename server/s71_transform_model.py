import sys
import os
import torch
import time

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
print('root_path--->', root_path)

my_path = root_path + '/src'
print('my_path--->', my_path)
sys.path.append(my_path) # 需要使用绝对路径  # 注意sys.path.append("./src") # 相对路径不起作用

# 如果pycharm显示不识别'红线错误'
# 解决方法：
#   （1）在项目 90_pgn_coverage_bm_server项目上，右键，
#   （2）弹出对话框中，Mark Directory as -> Sources Root

from src.m13_model import PGN
import src.m11_vocab as vocab
from src.m12_dataset import PairDataset
from src.d14_func_utils import timer
import src.d10_config as d10_config


'''
考虑GPU优化时, 会考虑两种情况:
第一种: GPU训练 + GPU部署
第二种: GPU训练 + CPU部署
'''

'''
假设只保存了模型的参数(model.state_dict())到文件名为 modelparameters.pth,  model = Net()

1 gpu -> cpu
    a1 = torch.load('modelparameters.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(a1)
    
2 cpu -> cpu或者gpu -> gpu
    a2 = torch.load('modelparameters.pth')
    model.load_state_dict(a2)

3 cpu -> gpu 1
    a3 = torch.load('modelparameters.pth', map_location=lambda storage, loc: storage.cuda(1))
    model.load_state_dict(a3)
    
    # 如果只有一个默认的设备
    a3 = torch.load('modelparameters.pth', map_location=lambda storage, loc: storage.cuda)
    model.load_state_dict(a3)

4 gpu 1 -> gpu 0
    a4 = torch.load('modelparameters.pth', map_location={'cuda:1':'cuda:0'})
    model.load_state_dict(a4)
'''

@timer(module='initalize and transform model...')
def transform_GPU_to_CPU(origin_model_path, to_device='cpu'):
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

    if to_device != 'cpu':
        print('Transform model to CPU!')
        exit(0)

    # 情况1 GPU训练 + CPU部署
    # model.load_state_dict( torch.load(origin_model_path, map_location = lambda storage, loc : storage) )

    # 情况2 GPU训练 + CPU部署
    my_load2 = torch.load(origin_model_path, map_location=torch.device('cpu'))
    # print('my_load2--->', my_load2)
    # model.load_state_dict(my_load2)
    # a = torch.device('cpu')
    # print('a--->', a, type(a))

    # 情况3 GPU训练 + CPU部署 将在GPU上训练好的模型加载到CPU上
    def my_func(storage, loc):
        # print('loc--->', loc)
        # print('storage--->', storage)
        return storage

    # my_load3 = torch.load(origin_model_path, map_location = my_func)
    # # print('my_load3-->', my_load3)
    # model.load_state_dict(my_load3)
    # print('model-->', model)

    # 情况4 gpu训练 gpu加载
    # 如果gpu训练cpu加载, 默认torch.load()会出错
    # 需要配置map_location参数torch.load(path ,map_location)
    model.load_state_dict(torch.load(origin_model_path))
    print(model)
    # model.to(to_device)


if __name__ == '__main__':

    transform_GPU_to_CPU(d10_config.model_save_path, 'cpu')
    print('GPU训练 + CPU部署 End')



  # 1 GPU训练 + GPU部署
    # model.load_state_dict(torch.load(origin_model_path))
    # print(model)
    # DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(DEVICE)
    # print('-------------------------------------------------------------------')

    '''
    RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.
     If you are running on a CPU-only machine, 
     please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.
    '''