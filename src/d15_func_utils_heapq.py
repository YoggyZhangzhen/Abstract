
import  random
import heapq

'''
堆的定义：
1 堆是一种特殊的数据结构，一个容器，里面存k个数据。根节点最大、最小。
  几万条数据，都扔给堆容器，堆容器可以筛选出k个最大值或者k个最小值
  小顶堆能把每个时间步的预测值进行筛选，保留k个最大值（eg：保留3个最大值）
  
2  python中heapq的使用
    列出一些常见的用法：
    heap = []               # 建立一个常见的堆
    heappush(heap,item)     # 往堆中插入一条新的值
    item = heappop(heap)    #弹出最小的值
    item = heap[0]          #查看堆中最小的值，不弹出
    heapify(x)              #将一个列表转为堆
    heappoppush()           #弹出最小的值，并且将新的值插入其中
    
    item = heapreplace(heap,item)   
                            #弹出一个最小的值，然后将item插入到堆当中。
                            堆的整体的结构不会发生改变。
                            
    merge()                 #将多个堆进行合并    
    nlargest(n , iterbale, key=None)
                            从堆中找出做大的N个数，key的作用和sorted()方法里面的key类似，
                            用列表元素的某个属性和函数作为关键字
'''

import math
import random
from io import StringIO

# 辅助函数：字符界面显示堆
def show_tree(tree, total_width=36, fill=' '):
    output = StringIO()
    last_row = -1
    for i, n in enumerate(tree):
        if i:
            row = int(math.floor(math.log(i + 1, 2)))
        else:
            row = 0
        if row != last_row:
            output.write('\n')
        columns = 2 ** row
        col_width = int(math.floor((total_width * 1.0) / columns))
        output.write(str(n).center(col_width, fill))
        last_row = row
    print (output.getvalue() )
    print ('-' * total_width )
    print()
    return


# 创建一个小顶堆
def dm01_heap_create():
    heap = [] # #建立一个常见的堆
    data = random.sample(range(1, 8), 7)
    print('data: ', data)

    for i in data:
        print('add %3d:' % i)
        heapq.heappush(heap, i)
        print(heap)
        show_tree(heap)

# heapq.heapify(list)
# 将list类型转化为heap, 重新排列列表
def dm02_heap_heapify():
    data =  [1, 6, 5, 2, 3, 4, 7]
    print(data)
    heapq.heapify(data)
    print(data)
    show_tree(data)


# 构建元素个数为 K=5 的最小堆代码实例:
def  dm03_heap_buildk5():
    myheap = []
    heapq.heapify(myheap)
    for i in range(15):
        item = random.randint(10, 100)
        print ("comeing ", item, end=' ')
        if len(myheap) >= 5:
            top_item = myheap[0]  # 取堆顶元素
            # 若新来的item比root值大，就把root弹出来，重新组织小顶堆
            if top_item < item:  # min heap
                top_item = heapq.heappop(myheap)
                print ("pop", top_item, end=' ')
                heapq.heappush(myheap, item)
                print ("push", item, myheap)
        else:
            heapq.heappush(myheap, item)
            print ("push", item, myheap)
            # print(myheap)

    print(myheap)
    print ("sort")
    myheap.sort()
    print (myheap)


# 小顶堆容器k=3
#   若来的数据item比root值大，就把root值弹出来，把item添加到小顶堆中
#   若来的数据item比root值小，不操作
def add2heap(heap, item, k):
    # Maintain a heap with k nodes and the smallest one as root.
    if len(heap) < k:
        heapq.heappush(heap, item)
    else:
        heapq.heappushpop(heap, item)

# 构建一个k = 3的小顶推
def dm04_heap_k3():
    myheap = []
    heapq.heapify(myheap)
    k = 3
    for i in range(15):
        item = random.randint(10, 100)
        print('coming item', item, end=' ')
        add2heap(myheap, item, k)
        print(myheap)



# 删除并返回堆中最小的元素, 通过heapify() 和heappop()来排序。
def dm05_heapq_outher():
    data = random.sample(range(1, 8), 7)
    print('data: ', data)
    heapq.heapify(data)
    show_tree(data)

    heap = []
    while data:
        i = heapq.heappop(data)
        print ('pop %3d:' % i)
        show_tree(data)
        heap.append(i)
    print ('heap: ', heap)


# 删除现有元素并将其替换为一个新值。
def dm06_heap_replace():
    data = random.sample(range(1, 8), 7)
    print ('data: ', data)
    heapq.heapify(data)
    show_tree(data)

    for n in [8, 9, 10]:
        smallest = heapq.heapreplace(data, n)
        print ('replace %2d with %2d:' % (smallest, n) )
        show_tree(data)
    return data

def dm07_heapreplace():
    # data = dm04_heapreplace()
    data = range(1, 6)
    l = heapq.nlargest(3, data)
    print (l)  # [5, 4, 3]

    s = heapq.nsmallest(3, data)
    print (s)  # [1, 2, 3]


if __name__ == '__main__':

    # 创建一个小顶堆
    # dm01_heap_create()

    # 将list类型转化为heap, 重新排列列表
    # dm02_heap_heapify()


    # 构建元素个数为 K=5 的最小堆代码实例:
    # dm03_heap_buildk5()

    # 构建一个k = 3的小顶推
    dm04_heap_k3()


    print('小顶堆测试 End')

