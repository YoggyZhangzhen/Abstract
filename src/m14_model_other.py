
import torch

'''
PyTorch数据发散（scatter/scatter_add）与聚集（Gather）操作

1 scatter_(input, dim, index, src)
  input.scatter_(dim, index, src)
      (a)将src中数据 (b)根据index中的索引 (c)按照dim的方向填进input中
      一个典型的用标量来修改张量的一个例子
      scatter() 一般可用来对标签进行 one-hot 编码

2 问题本质
    copy数据 需要指出 src[i][j] -> myc[m][n] i*j与m*n的关系通用通过label进行表达 所以变得晦涩难懂了！

3 拓展阅读
    https://zhuanlan.zhihu.com/p/158993858    # 有画图
    https://www.jianshu.com/p/6ce1ad89417f    # 
'''

def dm01_test_scatter():

    batch_size = 4  # 4个样本
    class_num = 10  # 10分类

    #10分类- 4个样本的类别标签Y
    label2 = torch.LongTensor(batch_size, 1).random_() % class_num
    print('\n10分类，4个样本的Y，label2--->\n',label2)

    label = torch.tensor([[6], [0],[2],[3]])
    print('\n10分类，4个样本的Y label--->\n', label)

    # 用200的值，根据label索引，按照dim=1(按照行) 去修改torch.zeros的值
    myc = torch.zeros(batch_size, class_num, dtype=torch.long)
    print('myc--->\n', myc)

    myc = torch.zeros(batch_size, class_num, dtype=torch.long).scatter_(1, label, 200)
    print(myc)

    # tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])


def dm02_test_scatter():
    batch_size = 4  # 4个样本
    class_num = 10  # 10分类

    # 10分类- 4个样本的类别标签Y
    label2 = torch.LongTensor(batch_size, 1).random_() % class_num
    print('\n10分类，4个样本的Y，label2--->\n', label2)

    label = torch.tensor([[6], [0], [2], [3]])
    print('\n10分类，4个样本的Y label--->\n', label)

    # 用200的值，根据label索引，按照dim=1(按照行) 去修改torch.zeros的值
    myc = torch.zeros(batch_size, class_num, dtype=torch.float32)
    print('myc--->\n', myc)

    myc = torch.zeros(batch_size, class_num, dtype=torch.float32).scatter_(1, label, 200)
    print(myc)

    # tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])

    x = torch.tensor([[0, 1, 2],
                      [1, 2, 3],
                      [2, 3, 4],
                      [4, 5, 6]])
    p_x = torch.tensor([[0.1, 0.1, 0.2],
                        [0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4],
                        [0.4, 0.5, 0.6]], dtype=torch.float32)

    # 把p_x的值，按照x的索引，累加到myc中
    final_distribution = myc.scatter_add_(dim=1, index=x, src=p_x)
    import numpy as np 
    np.set_printoptions(suppress=True)

    print('final_distribution--->\n')
    print('{:2s}'.format(repr(final_distribution.numpy())))
    # print('{:2f}'.format(final_distribution.numpy()) )

# 聚集操作
def dm03_test_gather():
    class_num = 10
    batch_size = 4
    label = torch.LongTensor(batch_size, 1).random_() % class_num

    label = torch.tensor([[6], [0],[3],[2]])
    print('label:', label)

    # 用1的值，根据label索引，按照dim=1行的方式，去修改torch.zeros的值
    myc = torch.zeros(batch_size, class_num).scatter_(1, label, 1)
    print(myc)

    # tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    #         [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])


    # 聚集操作：把myc的值 按照label指定的位置，copy给 myresult
    label = torch.tensor([[6], [0], [3], [2]])                 # 实验1
    # label = torch.tensor( [[0, 1], [1, 2], [2, 3], [3, 4]] ) # 实验2
    myresult = torch.gather(myc, 1, label)
    print('myresult--->', myresult)

    # 发散操作：用0.88的值， 根据[[2],[3]]的索引， 去修改torch.zeros的值
    z = torch.zeros(2, 4).scatter_(1, torch.LongTensor([[2], [3]]), 0.88)
    print("z---> \n", z)

    print('demo04_test_scatter End')


def dm04_test_torch_stack():
    # sample_losses = torch.sum(torch.stack(step_losses, 1), 1)
    step_losses = []
    a1 = torch.tensor([1], dtype=torch.float32)
    a2 = torch.tensor([2], dtype=torch.float32)
    a3 = torch.tensor([3], dtype=torch.float32)
    step_losses.append(a1)
    step_losses.append(a2)
    step_losses.append(a3)

    myb1 = torch.stack(step_losses, 1) # 1*3 torch.stack()函数 dim=1 会按照行的方向堆积元素
    myb2 = torch.stack(step_losses, 0) # 3*1

    print('\nmyb1--->', myb1, myb1.shape, type(myb1))
    print('\nmyb2--->', myb2, myb2.shape, type(myb2))

    sample_losses = torch.sum(myb1, 1)
    print('sample_losses--->', sample_losses)

def dm05_test_set():
    # self.stop_word = list(set([self.vocab[x.strip()] for x in open(d10_config.stop_word_file).readlines()]))

    # myset = set()
    # myfile = open(d10_config.stop_word_file)
    # for x in myfile.readlines():
    #     a = x.strip()
    #     b = self.vocab [a]
    #     myset.add(b)
    #
    #
    # mylist = list(myset)
    # print(mylist)
    pass


if __name__ == '__main__':

    # 发散操作
    # dm01_test_scatter()
    # dm02_test_scatter()

    # 聚焦操作
    # dm03_test_gather()

    # dm04_test_torch_stack()
    dm05_test_set()

    print('model_other End')