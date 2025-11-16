
# 导入工具包
import sys
import os
from collections import Counter
import torch
from torch.utils.data import Dataset

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from d14_func_utils import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from m11_vocab import Vocab
import d10_config
from torch.utils.data import DataLoader


# 创建 数据对 的 类
class PairDataset(object):
    def __init__(self, filename, tokenize=simple_tokenizer, max_enc_len=None, max_dec_len=None,
                 truncate_enc=False, truncate_dec=False):
        # truncate_enc： 编码数据，也即是原文，是否参与截断
        # truncate_dec： 解码数据，也就是摘要，是否参与截断
        # tokenize：工具包中的辅助函数入口地址。函数功能：文本按空格切分, 返回结果列表

        print("Reading dataset %s..." % filename, end=' ', flush=True)
        self.filename = filename
        self.pairs = []

        # 直接读取训练集数据文件， 切分成 编码器数据、解码器数据，并做长度的统一
        with open(filename, 'r', encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):

                # 在数据预处理阶段 已约定 X Y 之间的数据分隔符是 <SEP>
                pair = line.strip().split('<SEP>')
                if len(pair) != 2:
                    print("Line %d of %s is error formed." % (i, filename))
                    print(line)
                    continue

                # 前半部分数据是编码器数据，即article原始文本( source document)
                enc = tokenize(pair[0])
                if max_enc_len and len(enc) > max_enc_len:
                    if truncate_enc:
                        enc = enc[:max_enc_len]
                    else:
                        continue

                # 后半部分数据是解码器数据，即abstract摘要文本
                dec = tokenize(pair[1])
                if max_dec_len and len(dec) > max_dec_len:
                    if truncate_dec:
                        dec = dec[:max_dec_len]
                    else:
                        continue
                # 以 元组数据对 的格式，存储到列表中
                self.pairs.append((enc, dec))

        print("%d pairs." % len(self.pairs))

    # 构建 模型所需的 字典函数
    def build_vocab(self, embed_file=None):

        # 对读取的文件进行 单词计数 统计
        # 实例化Counter类 对象
        word_counts = Counter()

        # 调用工具类函数 统计每个单词 出现的次数，
        # 并以字典方式保存在 word_counts 对象中
        count_words(word_counts, [enc + dec for enc, dec in self.pairs])

        # 初始化字典类
        vocab = Vocab()
        # 如果有预训练的词向量就直接加载 如果没有则随着模型一起训练
        vocab.load_embeddings(embed_file)

        # 将计数结果写入字典类中
        for word, count in word_counts.most_common(d10_config.max_vocab_size):
            vocab.add_words([word])

        return vocab


# 直接为后续创建 DataLoader 提供服务的 数据集预处理类
# 功能：调用一下产生一个样本 调用1个批次数量 就产生一个批次的 样本
class SampleDataset(Dataset):
    def __init__(self, data_pair, vocab):
        # source 原始数据
        self.src_sents = [x[0] for x in data_pair]
        # target 摘要数据 目标数据
        self.trg_sents = [x[1] for x in data_pair]
        self.vocab = vocab
        self._len = len(data_pair)

    # 负责取元素的函数
    def __getitem__(self, index):

        # 调用工具函数获取输入x和OOV
        x, oov = source2ids(self.src_sents[index], self.vocab)

        # 按照模型的要求，自定义返回格式，共有6个字段，每个字段"个性化定制"
        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] + abstract2ids(self.trg_sents[index], self.vocab, oov) + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.trg_sents[index])
            }

    def __len__(self):
        return self._len


# 创建DataLoader时自定义的数据处理函数
def collate_fn(batch):

    # 按照最大程度限制 对张量进行填充0操作
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    # 对一个批次中的数据 按照x_len字段进行排序
    data_batch = sort_batch_by_len(batch)

    # 模型需要6个字段
    # 依次取得所需要的字段 作为构建DataLoader的返回数据
    x = data_batch['x']
    x_max_length = max([len(t) for t in x])
    y = data_batch['y']
    y_max_length = max([len(t) for t in y])

    OOV = data_batch['OOV']
    len_OOV = torch.tensor(data_batch['len_OOV'])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch['x_len'])
    y_len = torch.tensor(data_batch['y_len'])

    return x_padded, y_padded, x_len, y_len, OOV, len_OOV


if __name__ == "__main__":

    print('从配置文件中获取参数信息', d10_config.max_vocab_size)

    # 1 训练集 数据对
    train_dataset = PairDataset(d10_config.train_data_path,
                          max_enc_len=d10_config.max_enc_len,
                          max_dec_len=d10_config.max_dec_len,
                          truncate_enc=d10_config.truncate_enc,
                          truncate_dec=d10_config.truncate_dec)


    # 2 验证集 数据对
    val_dataset = PairDataset(d10_config.val_data_path,
                              max_enc_len=d10_config.max_enc_len,
                              max_dec_len=d10_config.max_dec_len,
                              truncate_enc=d10_config.truncate_enc,
                              truncate_dec=d10_config.truncate_dec)

    # # 创建模型的单词字典
    # vocab 有97733个 使用了20000个+4个特殊字符 其他的都是oov单词
    vocab = train_dataset.build_vocab(embed_file=d10_config.embed_file)
    print('vocab--->', vocab.size())


    vocab2 = val_dataset.build_vocab(embed_file=d10_config.embed_file)


    train_data = SampleDataset(train_dataset.pairs, vocab)
    val_data = SampleDataset(val_dataset.pairs, vocab)

    # 定义训练集的数据迭代器
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=d10_config.batch_size,
                                  # batch_size=1, # 1 2 8 16 测试
                                  shuffle=True,
                                  collate_fn=collate_fn)

    num_batches = len(train_dataloader)
    print('num_batches--->', num_batches)


    for batch, data in enumerate(train_dataloader):
        x, y, x_len, y_len, oov, len_oovs = data
        print('x->', x)
        print('y->', y)
        print('x_len->', x_len)
        print('y_len->', y_len)
        print('oov->', oov)
        print('len_oovs->', len_oovs)

        print('batch->', batch)
        break

    print('dataset End')
