import torch
import torch.nn as nn
import torch.optim as optim
import pickle


'''
# 1 pickle类功能介绍
    pickle 提供了一个简单的持久化功能。可以将对象以文件的形式存放在磁盘上。
    pickle模块只能在Python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
    pickle序列化后的数据，可读性差，人一般无法识别。
    
# 2 pickle类的Api函数介绍
pickle.dump(obj, file[, protocol])
    序列化对象，并将结果数据流写入到文件对象中。
    参数protocol是序列化模式，默认值为0，表示以文本的形式序列化。
    protocol的值还可以是1或2，表示以二进制的形式序列化。

pickle.load(file)
　　反序列化对象。将文件中的数据解析为一个Python对象。
注意：在load(file)的时候，要让python能够找到类的定义，否则会报错

'''

class Person:

    def __init__(self, n, a):
        self.name = n
        self.age = a

    def show(self):
        print ( self.name+"_"+str(self.age) )


# 测试序列化类
def dm01_test_pickle():
    aa = Person("zhangsan", 20)
    aa.show()
    f = open('./myperson.txt','wb')
    pickle.dump(aa, f, 0)
    f.close()

    # del Person
    f = open('myperson.txt','rb')
    bb = pickle.load(f)
    f.close()
    bb.show()


'''
神经网络的训练有时需要几天、几周、甚至几个月，为了在每次使用模型时避免高代价的重复训练，
我们就需要将模型序列化到磁盘中，使用的时候反序列化到内存中。

PyTorch 提供了两种保存模型的方法：
直接序列化模型对象
存储模型的网络参数
'''
class Model(nn.Module):

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size * 2)
        self.linear2 = nn.Linear(input_size * 2, output_size)

    def forward(self, inputs):
        inputs = self.linear1(inputs)
        output = self.linear2(inputs)
        return output


def dm02_test_model_saveandload():
    model = Model(128, 10)

    # 第一个参数: 存储的模型
    # 第二个参数: 存储的路径
    # 第三个参数: 使用的模块
    # 第四个参数: 存储的协议
    torch.save(model, './test_model_save1.bin', pickle_module=pickle, pickle_protocol=2)

    # 第一个参数: 加载的路径
    # 第二个参数: 模型加载的设备
    # 第三个参数: 加载的模块
    mymodel = torch.load('./test_model_save1.bin', map_location='cpu', pickle_module=pickle)
    print('加载的模型1', mymodel)


# 测试 存储模型的网络参数
def test01():
    model = Model(128, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 定义存储参数
    save_params = {
        'init_params': {
            'input_size': 128,
            'output_size': 10
        },
        'acc_score': 0.98,
        'avg_loss': 0.86,
        'iter_numbers': 100,
        'optim_params': optimizer.state_dict(),
        'model_params': model.state_dict()
    }

    # 存储模型参数
    torch.save(save_params, './model_params2.bin')


def test02():
    # 加载模型参数
    model_params = torch.load('./model_params2.bin')
    # 初始化模型
    model = Model(model_params['init_params']['input_size'], model_params['init_params']['output_size'])
    # 初始化优化器
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(model_params['optim_params'])
    # 显示其他参数
    print('迭代次数:', model_params['iter_numbers'])
    print('准确率:', model_params['acc_score'])
    print('平均损失:', model_params['avg_loss'])


def dm03_test_model_params():
    test01()
    test02()

# 存储模型的网络参数
if __name__ == '__main__':

    # 测试 序列化对象类
    dm01_test_pickle()

    # 测试 直接序列化模型对象
    # dm02_test_model_saveandload()

    # 测试 存储模型的网络参数
    # dm03_test_model_params()
    # test01()
    # test02()
    print('End')