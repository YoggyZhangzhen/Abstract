import pickle
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from tensorboardX import SummaryWriter

from m12_dataset import PairDataset
from m13_model import PGN
import d10_config
from m21_evaluate import evaluate
from m12_dataset import collate_fn, SampleDataset
from d14_func_utils import ScheduledSampler, config_info


def train(dataset, val_dataset, v, start_epoch=0): 
    DEVICE = d10_config.DEVICE

    # 实例化PGN类对象
    model = PGN(v)
    model.to(DEVICE)

    print("loading data......")
    train_data = SampleDataset(dataset.pairs, v)
    val_data = SampleDataset(val_dataset.pairs, v)

    print("initializing optimizer......")
    # 定义模型训练的优化器
    optimizer = optim.Adam(model.parameters(), lr=d10_config.learning_rate)

    # 定义训练集的数据迭代器
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=d10_config.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn)

    # 验证集上的损失值初始化为一个大整数
    # val_losses = np.inf
    val_losses = 100000000.0

    # 实例化SummaryWriter 为了服务于TensorboardX写日志的可视化工具
    writer = SummaryWriter(d10_config.log_path)

    # 求会运行多少个epoch
    num_epochs =  len(range(start_epoch, d10_config.epochs))

    # 训练阶段采用teacher_forcing的策略
    teacher_forcing = True
    print('teacher_forcing = {}'.format(teacher_forcing))

    # 根据配置文件config.py中的设置 对整个数据集进行一定轮次的迭代训练
    with tqdm(total=d10_config.epochs) as epoch_progress:
        for epoch in range(start_epoch, d10_config.epochs):

            # 每一个epoch之前打印模型训练的相关参数信息
            print(config_info(d10_config))

            # 初始化每一个batch损失值的存放列表
            batch_losses = []  # Get loss of each batch
            num_batches = len(train_dataloader)
            
            # 针对每一个epoch 按照batch_size读取数据迭代训练
            with tqdm(total=num_batches//100) as batch_progress:
                for batch, data in enumerate(tqdm(train_dataloader)):
                    x, y, x_len, y_len, oov, len_oovs = data
                    assert not np.any(np.isnan(x.numpy()))

                    # 如果配置GPU 则加速训练
                    if d10_config.is_cuda:  # Training with GPUs.
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        x_len = x_len.to(DEVICE)
                        len_oovs = len_oovs.to(DEVICE)

                    # 设置模型进入训练模型
                    model.train()

                    # 老三样中第一步： 梯度清零
                    optimizer.zero_grad()

                    # 利用模型进行训练 并返回损失值
                    loss = model(x, x_len, y,len_oovs, batch=batch,
                                 num_batches=num_batches,
                                 teacher_forcing=teacher_forcing)

                    batch_losses.append(loss.item())

                    # 老三样的第二步 反向传播
                    loss.backward()

                    # 为了防止梯度爆炸而进行梯度裁剪
                    # Do gradient clipping to prevent gradient explosion.
                    clip_grad_norm_(model.encoder.parameters(), d10_config.max_grad_norm)
                    clip_grad_norm_(model.decoder.parameters(), d10_config.max_grad_norm)
                    clip_grad_norm_(model.attention.parameters(), d10_config.max_grad_norm)

                    # 老三样中的 参数更新
                    optimizer.step()

                    # 每隔100个batch记录一下损失信息
                    if (batch % 100) == 0:
                        batch_progress.set_description(f'Epoch {epoch}')
                        batch_progress.set_postfix(Batch=batch, Loss=loss.item())
                        batch_progress.update()

                        # 向tensorboard中写入损失值信息
                        writer.add_scalar(f'Average loss for epoch {epoch}',
                                           np.mean(batch_losses),
                                           global_step=batch)

            # 将一个epoch中所有batch的平均损失值 作为这个epoch的损失值
            epoch_loss = np.mean(batch_losses)

            # 设置/修改进度条的提示，字符串后自动添加 “:”
            epoch_progress.set_description(f'Epoch {epoch}')
            # 设置/修改进度条后的提示信息
            epoch_progress.set_postfix(Loss=epoch_loss)
            epoch_progress.update()

            # 结束每一个epoch训练后，直接在验证集上跑一下模型的效果
            avg_val_loss = evaluate(model, val_data, epoch)

            print('training loss:{}'.format(epoch_loss), 'validation loss:{}'.format(avg_val_loss))

            # 更新一下有更小损失值的模型
            if (avg_val_loss < val_losses):
                torch.save( model.state_dict(), d10_config.model_save_path)
                val_losses = avg_val_loss

                # 将更小的损失值写入文件中保存
                with open(d10_config.losses_path, 'wb') as f:
                    pickle.dump(val_losses, f) # pickle.dump() pickle.load() 实现基本数据类型序列化和反序列化

    # 关闭可视化写对象
    writer.close()



if __name__ == "__main__":
    # Prepare dataset for training.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('DEVICE: ', DEVICE)

    # 构建训练用的数据集
    dataset = PairDataset(d10_config.train_data_path,
                          max_enc_len=d10_config.max_enc_len,
                          max_dec_len=d10_config.max_dec_len,
                          truncate_enc=d10_config.truncate_enc,
                          truncate_dec=d10_config.truncate_dec)

    # 构建测试用的数据集
    val_dataset = PairDataset(d10_config.val_data_path,
                              max_enc_len=d10_config.max_enc_len,
                              max_dec_len=d10_config.max_dec_len,
                              truncate_enc=d10_config.truncate_enc,
                              truncate_dec=d10_config.truncate_dec)

    # vocab 20004个
    # 创建模型的单词字典
    vocab = dataset.build_vocab(embed_file=d10_config.embed_file)

    # 调用训练函数进行模型训练
    train(dataset, val_dataset, vocab, start_epoch=0)







