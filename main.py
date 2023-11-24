import torch
import torch.nn.functional as F
import math
import argparse  # 用于解析命令行参数
import numpy as np
import os


from DSAN import DSAN
import data_loader


def load_data(root_path, src, tar, batch_size):
    """
    加载数据集
    Args:
        root_path (字符串): 数据集的根目录
        src (字符串): 源域的名称
        tar (字符串): 目标域的名称
        batch_size (int): 批处理的大小
    """
    kwargs = {'num_workers':1, 'pin_memory':True}
    # 使用data_loader中的load_training函数，加载源域核目标域的训练数据，返回两个数据加载器

    '''
    print('root_path:', root_path)
    print('src:', src)
    打印结果
    root_path: /home/zxd/桌面/Deep_Learning/SEI_data/8_month
    src: 20230817
    '''

    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(
        root_path, tar, batch_size, kwargs
    )
    return loader_src, loader_tar, loader_tar_test


def train_epoch(epoch, model, dataloaders, optimizer):
    """
    定义训练一个周期的函数
    Args:
        epoch (int): 当前的周期数
        model (DSAN类的实例): 待训练的模型
        dataloaders (tuple): 包含三个数据加载器的元组
        optimizer (torch.optim类的实例): 用于优化模型的优化器
    """
    model.train()
    # print(dataloaders) -> (None, None, None)
    # TypeError: 'NoneType' object is not iterable->第一次报错
    source_loader, target_train_loader, _ = dataloaders
    # 分别创建源域核目标域的训练数据的迭代器，用于按批次获取数据
    iter_source = iter(source_loader)
    iter_target = iter(target_train_loader)
    # 获取源域的训练数据的批次数，作为一个周期的迭代次数
    num_iter = len(source_loader)

    for i in range(1, num_iter):
        # 从源域的训练数据的迭代器中获取一批数据和标签
        data_source, label_source = iter_source.next()
        # 从目标域的训练数据的迭代器中获取一批数据，忽略标签
        data_target, _ = iter_target.next()
        # 如果目标域的训练数据的迭代器已经遍历完毕，就重新创建一个
        if i % len(target_train_loader) == 0:
            iter_target = iter(target_train_loader)
        # 将源域的数据和标签转移到GPU上
        data_source, label_source = data_source.cuda(), label_source.cuda()
        data_target = data_target.cuda()

        # 清空优化器的梯度缓存
        optimizer.zero_grad()
        # 将源域和目标域 的数据和源域标签输入到模型中，得到源域的分类结果和自适应损失
        label_source_pred, loss_lmmd = model(
            data_source, data_target, label_source
        )
        # 使用负对数似然损失函数，计算源域的分类损失
        loss_cls = F.nll_loss(F.log_softmax(
            label_source_pred, dim=1), label_source)
        # 根据当前的周期数，计算一个动态的权重系数，用于平衡分类损失和自适应损失
        lambd = 2 / (1 + math.exp(-10 * (epoch) / args.nepoch)) - 1
        # 总损失=分类损失+自适应损失
        loss = loss_cls + args.weight * lambd * loss_lmmd

        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        # 打印当前周期数，损失，分类损失和自适应损失
        if i % args.log_interval == 0:
            print(f'Epoch: [{epoch:2d}], Loss: {loss.item():.4f}, cls_Loss: {loss_cls.item():.4f}, loss_lmmd: {loss_lmmd.item():.4f}')

def test(model, dataloader):
    """
    测试
    Args:
        model (DSAN类的实例): 待测试的模型
        dataloader (torch.utils.data.DataLoader类的实例): 用于测试的数据加载器
    """
    model.eval()
    test_loss = 0  # 测试损失
    correct = 0  # 正确分类的数量
    with torch.no_grad():
        # 对于数据加载器中的每一批数据和标签
        for data, target in dataloader:
            data, target = data.cuda(), target.cuda()
            # 使用模型的预测函数，对数据进行分类
            # 得到一个二维张量，形状为(batch_size, num_classes)
            # 每一行是一个样本的类别概率分布
            pred = model.predict(data)
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            # 从类别概率分布中，取出最大值对应的索引，作为预测的类别
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        # 将测试损失除以数据加载器的长度，得到平均测试损失
        test_loss /= len(dataloader)
        # 打印出平均测试损失和准确率
        print(
            f'Average loss: {test_loss:.4f}, Accuracy: {correct} / {len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)'
        )
    # 返回正确分类的数量，用于判断是否提前停止训练
    return correct


def get_args():
    # 用于解析一些布尔类型的参数
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')
    

    # 解析命令行参数的解析器
    parser = argparse.ArgumentParser()
    # 添加一个参数，表示数据集的根目录，一个字符串，为'/home/zxd/桌面/Deep_Learning/SEI_data/8_month'
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                         default='/home/zxd/桌面/Deep_Learning/dataset')
    # 添加一个参数，表示源域名称，一个字符串，为‘20230817‘
    parser.add_argument('--src', type=str, help='Source domain', default='data20230531')
    # 添加一个参数，表示目标域的名称，一个字符串,为‘20230818’
    parser.add_argument('--tar', type=str, help='Source domain',
                         default='data20230601')
    # 添加一个参数，表示子领域的数量，即类别数量，为8
    parser.add_argument('--nclass', type=int,
                         help='Number of classes', default=8)
    # 添加一个参数，表示batch_size=32
    parser.add_argument('--batch_size', type=float, help='batch_size',
                         default=32)
    # 总的周期数，为50
    parser.add_argument('--nepoch', type=int, help='Total epoch num',
                         default=50)
    # 添加学习率，一个列表，包含三个浮点数[0.0001, 0.001, 0.01]
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.0001, 0.001, 0.001])
    # 提前停止的次数
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=300)
    # 随机种子
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    # 自适应损失的权重系数
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=0.5)
    # 动量，用于梯度下降
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    # 权重衰减
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=200)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    args = get_args()  # 获取命令行参数
    print(vars(args))
    SEED = args.seed  # 设置随机种子，保证实验的可重复性
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataloaders = load_data(args.root_path, args.src,
                            args.tar, args.batch_size)
    # print(dataloaders)
    model = DSAN(num_classes=args.nclass).cuda()

    # print(model)
    
    correct = 0
    stop = 0

    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.bottle.parameters(), 'lr': args.lr[1]},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
    else:
        optimizer = torch.optim.SGD([
            {'params': model.feature_layers.parameters()},
            {'params': model.cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)

    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)

        train_epoch(epoch, model, dataloaders, optimizer)
        t_correct = test(model, dataloaders[-1])
        if t_correct > correct:
            correct = t_correct
            stop = 0
            torch.save(model, 'model.pkl')
        print(
            f'{args.src}-{args.tar}: max correct: {correct} max accuracy: {100. * correct / len(dataloaders[-1].dataset):.2f}%\n')

        if stop >= args.early_stop:
            print(
                f'Final test acc: {100. * correct / len(dataloaders[-1].dataset):.2f}%')
            break

