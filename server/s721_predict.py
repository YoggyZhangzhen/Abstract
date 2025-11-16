import random
import os
import sys
import torch
import jieba

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

my_path = root_path + '/src'
print('my_path--->', my_path)
sys.path.append(my_path) # 需要使用绝对路径  # 注意sys.path.append("./src") # 相对路径不起作用


import src.d10_config as d10_config
from src.m13_model import PGN
from src.m12_dataset import PairDataset
from src.d14_func_utils import source2ids, outputids2words, Beam, timer, add2heap, replace_oovs

# 创建预测类
class Predict():

    # @timer(module='initalize predicter')
    def __init__(self):
        self.DEVICE = d10_config.DEVICE

        dataset = PairDataset(d10_config.train_data_path,
                              max_enc_len=d10_config.max_enc_len,
                              max_dec_len=d10_config.max_dec_len,
                              truncate_enc=d10_config.truncate_enc,
                              truncate_dec=d10_config.truncate_dec)

        self.vocab = dataset.build_vocab(embed_file=d10_config.embed_file)

        self.model = PGN(self.vocab)
        self.stop_word = list(set([self.vocab[x.strip()] for x in open(d10_config.stop_word_file).readlines()]))

        self.model.load_state_dict(torch.load(d10_config.model_save_path,
                                              map_location=lambda storage, loc:storage),
                                            False)

        # 导入已经训练好的模型 并转移到GPU上
        # self.model.load_state_dict(torch.load(config.model_save_path), map_location=torch.device('cpu'))

        self.model.to(self.DEVICE)



    # 编写贪心解码的类内函数
    def greedy_search(self, x, max_sum_len, len_oovs, x_padding_masks):
        # Get encoder output and states.Call encoder forward propagation
        # 模型编码
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))
        # 用encode的hidden state来初始化 decoder中的 hidden state
        decoder_states = self.model.reduce_state(encoder_states)

        # Initialize decoder's input at time step 0 with the SOS token.
        # 利用 sos 作为解码器的初始化输入字符
        x_t = torch.ones(1) * self.vocab.SOS
        x_t = x_t.to(self.DEVICE, dtype=torch.int64)
        summary = [self.vocab.SOS] # 初始化摘要 用SOS作为起始
        # 增加 coverage机制的代码

        coverage_vector = torch.zeros( (1, x.shape[1])).to(self.DEVICE)

        # Generate hypothesis with maximum decode step.
        # 循环解码 最多解码max_sum_len步
        while int(x_t.item()) != (self.vocab.EOS) and len(summary) < max_sum_len:
            context_vector, attention_weights, next_coverage_vector = self.model.attention(decoder_states, encoder_output, x_padding_masks, coverage_vector)
            p_vocab, decoder_states, p_gen = self.model.decoder(x_t.unsqueeze(1), decoder_states, context_vector)
            final_dist = self.model.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))
            next_coverage_vector = context_vector  # 增加coverage机制
            # Get next token with maximum probability.
            # 以贪心解码的策略进行预测字符
            x_t = torch.argmax(final_dist, dim=1).to(self.DEVICE)
            decoder_word_idx = x_t.item()

            # 预测的字符添加进结果摘要中
            summary.append(decoder_word_idx)
            x_t = replace_oovs(x_t, self.vocab)

        return summary

    # -----------------------------------------------------------------------------------------
    # 下面的函数best_k()和beam_search()是6.2小节新增的, 用于支持Beam-search解码的新函数.


    # 编写实现生成前k个最大概率token的函数

    # beam: 代表Beam类的一个实例化对象   # 假若 x[181,], x_padding_masks[181,], len_oovs[3,], encoder_output[1,181,1024]
    # k: 代表Beam-search中的重要参数beam_size=k.
    # encoder_output: 编码器的输出张量.
    # x_padding_masks: 输入序列的padding mask, 用于遮掩那些无效的PAD位置字符
    # x: 编码器的输入张量
    # len_oovs: OOV列表的长度

    # 根据搜索宽度求预测值top-k
    def best_k(self, beam, k, encoder_output, x_padding_masks, x, len_oovs):

        # 当前时间步t的对应解析字符token, 将作为Decoder端的输入, 产生最终的vocab distribution.
        x_t = torch.tensor(beam.tokens[-1]).reshape(1, 1)
        x_t = x_t.to(self.DEVICE)

        # 通过注意力层attention, 得到context_vector  [(1,1,512),(1,1,512)], (1,181,1024),[135,],[1,135] -> [1,1024],[1,181],[1,181]
        context_vector, attention_weights, coverage_vector = self.model.attention(beam.decoder_states, encoder_output, x_padding_masks, beam.coverage_vector)
        # 函数replace_oovs()将OOV单词替换成新的id值, 来避免解码器出现index-out-of-bound error [1,1], [(1,1,512),(1,1,512)],[1,1024] -->[1,20004], [(1,1,512),(1,1,512)], [1,1]
        p_vocab, decoder_states, p_gen = self.model.decoder(replace_oovs(x_t, self.vocab), beam.decoder_states, context_vector)

        # 调用PGN网络中的函数, 得到最终的单词分布(包含OOV) final_dist[1,20004+len(oov)] final_dist[1,20009]
        final_dist = self.model.get_final_distribution(x, p_gen, p_vocab, attention_weights, torch.max(len_oovs))

        # 计算序列的log_probs分数
        log_probs = torch.log(final_dist.squeeze()) # log_probs[20009,]
        # 如果当前Beam序列只有1个token, 要将一些无效字符删除掉, 以免影响序列的计算.
        # 至于这个无效字符的列表都包含什么, 也是利用bad case的分析, 结合数据观察得到的, 属于调优的一部分
        if len(beam.tokens) == 1:
            forbidden_ids = [self.vocab[u"这"],
                             self.vocab[u"此"],
                             self.vocab[u"采用"],
                             self.vocab[u"，"],
                             self.vocab[u"。"]]

            log_probs[forbidden_ids] = -float('inf')

        # 对于EOS token的一个罚分处理.
        # 具体做法参考了 https://opennmt.net/OpenNMT/translation/beam_search/.
        log_probs[self.vocab.EOS] *= d10_config.gamma * x.size()[1] / len(beam.tokens)
        log_probs[self.vocab.UNK] = -float('inf')

        # 从log_probs中获取top_k分数的tokens, 这也正好符合beam-search的逻辑.
        topk_probs, topk_idx = torch.topk(log_probs, k) # torch.topk((20007,), 3) -> topk_probs[3,], topk_idx[3,]

        # 非常关键的一行代码: 利用top_k的单词, 来扩展beam-search搜索序列, 等效于将top_k单词追加到候选序列的末尾.
        best_k = [beam.extend(x, log_probs[x], decoder_states, coverage_vector) for x in topk_idx.tolist()]

        # 返回追加后的结果列表
        return best_k

    def beam_search(self, x, max_sum_len, beam_size, len_oovs, x_padding_masks):

        # x: 编码器的输入张量, 即 article(source document)
        # max_sum_len: 本质上就是最大解码长度 max_dec_len
        # beam_size: 采用beam-search策略下的搜索宽度k
        # len_oovs: OOV列表的长度
        # x_padding_masks: 针对编码器的掩码张量, 把无效的PAD字符遮掩掉

        # 第一步: 通过Encoder计算得到编码器的输出张量. [1,181]-> [1,181,1024] ([2,1,512],[2,1,512])
        encoder_output, encoder_states = self.model.encoder(replace_oovs(x, self.vocab))
        # 对encoder_states进行加和降维处理, 赋值给decoder_states.([2,1,512],[2,1,512]) -> ([1,1,512],[1,1,512])
        decoder_states = self.model.reduce_state(encoder_states)

        # 全零张量初始化coverage vector [1,181]
        coverage_vector = torch.zeros((1, x.shape[1])).to(self.DEVICE)


        # 初始化 hypothesis [猜测列表] , 第一个token给SOS, 分数给0.
        init_beam = Beam([self.vocab.SOS], [0], decoder_states, coverage_vector)

        # beam_size本质上就是搜索宽度k
        k = beam_size
        # 初始化 curr 作为当前候选集, completed 作为最终的hypothesis列表
        curr, completed = [init_beam], []

        # 通过for循环连续解码 max_sum_len 步, 每一步应用 beam-search策略产生预测token.
        for _ in range(max_sum_len):
            # 初始化当前时间步的 topk列表 为空, 后续将beam-search的解码结果存储在 topk列表 中.
            topk = []
            # print('第%d个时间步_'  %_)
            # print("第{}个时间步".format(_))
            for beam in curr:
                # 如果产生了一个EOS token, 则将beam对象追加进最终的hypothesis列表, 并将k值减1, 然后继续搜索.
                if beam.tokens[-1] == self.vocab.EOS:
                    completed.append(beam)
                    k -= 1
                    continue

                # 遍历最好的k个 候选集序列 (也就是，根据搜索宽度求预测值top-k)
                my_best_k_list = self.best_k(beam, k, encoder_output, x_padding_masks, x, torch.max(len_oovs) )
                for can in my_best_k_list:
                    # 利用小顶堆来维护一个top_k的candidates.
                    # 小顶堆的值以当前序列的得分为准, 顺便也把候选集的id和候选集本身存储起来.
                    # tmpseq = can.seq_score()
                    # tmpid = id(can)
                    # tmpcan = can
                    # print(tmpseq, tmpid, tmpcan)
                    add2heap(topk, (can.seq_score(), id(can), can), k)

            # 当前候选集是堆元素的index=2的值can.
            curr = [items[2] for items in topk]
            # 候选集数量已经达到搜索宽度的时候, 停止搜索.
            if len(completed) == beam_size:
                break

        # 将最后产生的候选集追加进completed中.
        completed += curr

        # 按照得分进行降序排列, 取分数最高的作为当前解码结果序列.
        result = sorted(completed, key=lambda x: x.seq_score(), reverse=True)[0].tokens
        return result

    # 上面的两个函数best_k()和beam_search()是6.2小节为了支持Beam-search解码的新函数
    # --------------------------------------------------------------------------------------


    # 编写预测函数的代码
    @timer(module='doing prediction')
    def predict(self, text, tokenize=True, beam_search=False):

        if isinstance(text, str) and tokenize:
            text = list(jieba.cut(text))

        # 将原始文本映射成数字化张量
        x, oov = source2ids(text, self.vocab)
        x = torch.tensor(x).to(self.DEVICE)

        # 获取oov的长度 和 paddiing_mask张量
        len_oovs = torch.tensor([len(oov)]).to(self.DEVICE)
        x_padding_masks = torch.ne(x, 0).byte().float()

        # ------------------------------------------------------------------------
        # 下面是6.2小节的新增代码部分, 采用beam search策略进行解码
        if beam_search:
            summary = self.beam_search(x.unsqueeze(0),
                                       max_sum_len = d10_config.max_dec_steps,
                                       beam_size = d10_config.beam_size,
                                       len_oovs = len_oovs,
                                       x_padding_masks = x_padding_masks)
        # ------------------------------------------------------------------------

        # 采用贪心策略进行解码
        else:
            summary = self.greedy_search(x.unsqueeze(0),
                                         max_sum_len = d10_config.max_dec_steps,
                                         len_oovs = len_oovs,
                                         x_padding_masks = x_padding_masks)

        # 将得到的摘要数字化张量 转换成自然语言文本
        summary = outputids2words(summary, oov, self.vocab)

        # 删除掉特殊字符 SOS EOS ，去除空字符
        return summary.replace('<SOS>', '').replace('<EOS>', '').strip()


# if __name__ == "__main__":
#     print('实例化Predict对象, 构建 dataset 和 vocab ......')
#     pred = Predict()
#     print('vocab_size: ', len(pred.vocab))
#
#     # Randomly pick a sample in test set to predict.
#     # 随机 从测试集中抽取一条样本进行预测
#     with open(d10_config.val_data_path, 'r') as test:
#         picked = random.choice(list(test))
#         source, ref = picked.strip().split('<SEP>')
#
#     print('原始文本source: ', source, '\n')
#     print('******************************************')
#     print('人工智能摘要ref: ', ref, '\n')
#     print('******************************************')
#
#     greedy_prediction = pred.predict(source.split())
#     print('预测摘要: ', greedy_prediction, '\n')



if __name__ == "__main__":

    print('实例化Predict对象, 构建dataset和vocab......')
    pred = Predict()
    print('vocab_size: ', len(pred.vocab))

    # Randomly pick a sample in test set to predict.
    with open(d10_config.val_data_path, 'r') as test:
        picked = random.choice(list(test))
        source, ref = picked.strip().split('<SEP>')

    print('source: ', source, '\n')
    print('******************************************')
    print('ref: ', ref, '\n')
    print('******************************************')

    # greedy_prediction = pred.predict(source.split(),  beam_search=False)
    # print('greedy: ', greedy_prediction, '\n')

    print('******************************************')
    beam_prediction = pred.predict(source.split(),  beam_search=True)
    print('beam: ', beam_prediction, '\n')

