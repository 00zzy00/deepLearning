import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from torchvision import transforms
from torch.utils.data import DataLoader

from models.baseline import BaselineModel
from models.maniqa import IQA
from config import Config
from utils.process import RandCrop, ToTensor, Normalize, five_point_crop, split_dataset_scid
from utils.process import split_dataset_kadid10k, split_dataset_koniq10k
from utils.process import RandRotation, RandHorizontalFlip
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""确保在深度学习应用中的随机性操作（例如权重初始化、数据增强等）在不同运行之间具有一致的结果。
   设置随机数种子"""


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # 设置PyTorch的CuDNN库为确定性模式，以确保每次运行时的结果一致性，尽管这可能会降低一些性能。
    torch.backends.cudnn.deterministic = True


"""设置日志记录（logging）的函数"""


def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    #  创建日志路径
    # 构建日志文件路径，是日志文件名和日志路径的组合
    filename = os.path.join(config.log_path, config.log_file)
    # 用于配置日志记录
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


"""训练循环函数：
    属性：训练轮数、网络模型、损失函数、优化器、学习率调度器、训练数据加载器"""


def train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader):
    # 用于存储每个小批次的损失值，以便后续计算平均损失。
    losses = []
    # 这一行将神经网络模型切换到训练模式，
    # 这通常用于启用一些训练特定的操作，例如批量标准化的更新。
    net.train()
    # save data for one epoch
    # 用于保存每个小批次的模型预测和真实标签
    pred_epoch = []
    labels_epoch = []

    # 循环迭代train_loader训练数据
    for data in tqdm(train_loader):
        # 从数据中提取原始图像（d_img_org），并将其移到GPU上，
        # 以便进行神经网络的前向传播计算
        x_d = data['d_img_org'].cuda()
        # 提取目标标签数据
        labels = data['score']

        # 将标签数据转换为浮点类型，并确保其在GPU上。
        # 同时，通过torch.squeeze函数去除标签中的不必要的维度，以便后续计算。
        labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
        # 将前向传播预测结果赋值到pre_d
        pred_d = net(x_d)
        # 首先将梯度归零
        optimizer.zero_grad()
        # 传递真实的主观评价值和客观的预测值，计算损失函数
        loss = criterion(torch.squeeze(pred_d), labels)
        # 将每一个epoch的损失函数值拼接到列表中
        losses.append(loss.item())
        # 根据计算的损失函数完成
        loss.backward()
        # 使用优化器来更新模型的参数，以减小损失
        optimizer.step()
        # 优化学习率
        scheduler.step()

        # save results in one epoch
        # 将该预测结果转换为numpy放入CPU中计算
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)

    # compute correlation coefficient
    # 计算Srcc \Plcc
    rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
    # 计算所有批次的平均损失
    ret_loss = np.mean(losses)
    # 记录训练进度，包括当前轮数、平均损失、Spearman相关性系数（SRCC）和Pearson相关性系数（PLCC）。保留小数点后四位
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4} '.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p


"""评估（测试）神经网络模型的一个评估循环（evaluation loop）的函数:
    传入参数：配置信息、训练轮数、网络模型、损失函数、测试数据加载器"""


def eval_epoch(config, epoch, net, criterion, test_loader):
    # 表明在测试阶段不会更新和计算梯度
    with torch.no_grad():
        losses = []
        # 将神经网络模型切换到评估模式，这通常用于禁用一些训练特定的操作，
        # 例如批量标准化的更新
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        # 用于存储每个 epoch 的 PLCC 和 SROCC 值：
        test_plcc_values = []
        test_srocc_values = []

        for data in tqdm(test_loader):
            pred = 0
            # 配置中指定评估次数num_avg_val，属于超参数
            for i in range(config.num_avg_val):
                x_d = data['d_img_org'].cuda()
                labels = data['score']
                labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                # 在测试阶段对图像进行检测变换以减少过拟合的风险
                x_d = five_point_crop(i, d_img=x_d, config=config)
                # 获取数据并进行前向传播，累积多次的预测结果。
                pred += net(x_d)
            # 计算多次预测的平均值，以获得最终的预测结果。
            pred /= config.num_avg_val
            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s,
                                                                                 rho_p))
        return np.mean(losses), rho_s, rho_p

if __name__ == '__main__':
    # 使用的CPU线程数量
    cpu_num = 1
    # 设置了一系列环境变量，控制并行计算库在CPU上使用的线程数量，从而影响计算性能。
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # config file
    config = Config({
        # dataset path
        "dataset_name": 'siqad',

        # PIPAL
        "train_dis_path": "./PIPAL22/Train_dis/",
        "val_dis_path": "./PIPAL22/Val_dis/",
        "pipal22_train_label": "./data/PIPAL22/pipal22_train.txt",
        "pipal22_val_txt_label": "./data/PIPAL22/pipal22_val.txt",

        # KADID-10K
        "kadid10k_path": "./kadid10k/kadid10k/images/",
        "kadid10k_label": "./data/kadid10k/kadid10k_label.txt",

        # KONIQ-10K
        "koniq10k_path": "./koniq10k_1024x768/1024x768/",
        "koniq10k_label": "./data/koniq10k/koniq10k_label.txt",

        # SCID
        "scid_path": "./SCID/DistortedSCIs/",
        "scid_label": "./data/scid/scid_label.txt",

        # SIQAD
        "siqad_path": "./SIQAD/DistortedImages/",
        "siqad_label": "./data/SIQAD/siqad_label.txt",

        # optimization
        "batch_size": 2,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 200,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1,  # if training koniq10k, num_avg_val is set to 1
        "num_workers": 0,

        # data
        "split_seed": 20,
        "train_keep_ratio": 1.0,
        "val_keep_ratio": 1.0,
        "crop_size": 224,
        "prob_aug": 0.7,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "num_S2a": 2,
        "scale": 0.8,

        # load & save checkpoint
        "model_name": "koniq10k-base_s20",
        "type_name": "Koniq10k",
        "ckpt_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/"
    })
    # 设置日志文件的名称，通常使用模型名称作为基础名称，附加 ".log" 扩展名。
    """这段代码主要用于配置和准备日志记录和 TensorBoard 日志，
    包括设置日志文件名、创建相关目录、记录配置信息和创建 TensorBoard 日志写入器。
    这些操作有助于跟踪训练和评估的进展以及模型性能。"""
    config.log_file = config.model_name + ".log"

    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name)
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.model_name)

    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)

    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    if config.dataset_name == 'kadid10k':
        from data.kadid10k.kadid10k import Kadid10k

        train_name, val_name = split_dataset_kadid10k(
            txt_file_name=config.kadid10k_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.kadid10k_path
        dis_val_path = config.kadid10k_path
        label_train_path = config.kadid10k_label
        label_val_path = config.kadid10k_label
        Dataset = Kadid10k
    elif config.dataset_name == 'pipal':
        from data.PIPAL22.pipal import PIPAL

        dis_train_path = config.train_dis_path
        dis_val_path = config.val_dis_path
        label_train_path = config.pipal22_train_label
        label_val_path = config.pipal22_val_txt_label
        Dataset = PIPAL
    elif config.dataset_name == 'koniq10k':
        from data.koniq10k.koniq10k import Koniq10k

        train_name, val_name = split_dataset_koniq10k(
            txt_file_name=config.koniq10k_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.koniq10k_path
        dis_val_path = config.koniq10k_path
        label_train_path = config.koniq10k_label
        label_val_path = config.koniq10k_label
        Dataset = Koniq10k
    elif config.dataset_name == 'scid':
        from data.scid.scid import Scid

        train_name, val_name = split_dataset_scid(
            txt_file_name=config.scid_label,
            split_seed=config.split_seed
        )
        dis_train_path = config.scid_path
        dis_val_path = config.scid_path
        label_train_path = config.scid_label
        label_val_path = config.scid_label
        Dataset = Scid
    elif config.dataset_name == 'siqad':
        from data.SIQAD.siqad import Siqad

        train_name, val_name = split_dataset_scid(
            txt_file_name=config.siqad_label,
            split_seed=config.siqad_seed
        )
        dis_train_path = config.siqad_path
        dis_val_path = config.siqad_path
        label_train_path = config.siqad_label
        label_val_path = config.siqad_label
        Dataset = Siqad
    else:
        pass

    # data load:两行分别创建了训练数据集和验证数据集的实例，
    # 并将它们分别赋值给 train_dataset 和 val_dataset 变量。
    # transform - 这是一个数据变换（数据增强）的序列，它定义了在加载数据时应用的一系列变换操作。这些变换可以包括随机裁剪(
    # RandCrop)、归一化(Normalize)、随机水平翻转(RandHorizontalFlip)、转换为张量(ToTensor)
    # 等。这些变换有助于在训练期间增强数据，以提高模型的性能和泛化能力。
    train_dataset = Dataset(
        dis_path=dis_train_path,
        txt_file_name=label_train_path,
        list_name=train_name,
        transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
                                      Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=config.prob_aug), ToTensor()]),
        keep_ratio=config.train_keep_ratio
    )
    val_dataset = Dataset(
        dis_path=dis_val_path,
        txt_file_name=label_val_path,
        list_name=val_name,
        transform=transforms.Compose([RandCrop(patch_size=config.crop_size),
                                      Normalize(0.5, 0.5), ToTensor()]),
        keep_ratio=config.val_keep_ratio
        # 控制是否保持数据集中的样本比例
    )

    logging.info('number of train scenes: {}'.format(len(train_dataset)))
    logging.info('number of val scenes: {}'.format(len(val_dataset)))

    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              num_workers=config.num_workers, drop_last=True, shuffle=True)

    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            num_workers=config.num_workers, drop_last=True, shuffle=False)

    # model defination
    net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
                 patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
                 depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)
    # net = BaselineModel()

    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    net = nn.DataParallel(net)
    net = net.cuda()

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    # train & validation
    # 初始化一些变量，如 losses 和 scores，
    # 用于存储损失和分数，以及一些用于跟踪最佳性能指标的变量，
    # 如 best_srocc 和 best_plcc。
    losses, scores = [], []
    best_srocc = 0
    best_plcc = 0
    main_score = 0

    # 用于存储每个 epoch 的 PLCC 和 SROCC 值：
    test_plcc_values = []
    test_srocc_values = []
    for epoch in range(0, config.n_epoch):
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))
        loss_val, rho_s, rho_p = train_epoch(epoch, net, criterion, optimizer, scheduler, train_loader)

        writer.add_scalar("Train_loss", loss_val, epoch)
        writer.add_scalar("SRCC", rho_s, epoch)
        writer.add_scalar("PLCC", rho_p, epoch)

        losses.append(loss_val)
        scores.append(rho_s + rho_p)


        if (epoch + 1) % config.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running testing in epoch {}'.format(epoch + 1))
            loss, rho_s, rho_p = eval_epoch(config, epoch, net, criterion, val_loader)
            test_plcc_values.append(rho_p)
            test_srocc_values.append(rho_s)
            logging.info('Eval done...')

            if rho_s + rho_p > main_score:
                main_score = rho_s + rho_p
                best_srocc = rho_s
                best_plcc = rho_p

                logging.info('======================================================================================')
                logging.info(
                    '============================== best main score is {} ================================='.format(
                        main_score))
                logging.info('======================================================================================')

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(config.ckpt_path, model_name)
                torch.save(net.module.state_dict(), model_save_path)
                logging.info(
                    'Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc, best_plcc))

        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

    epochs = range(1, config.n_epochs + 1)
    plt.plot(epochs, test_plcc_values, label='Test PLCC')
    plt.plot(epochs, test_srocc_values, label='Test SROCC')
    plt.xlabel('Epochs')
    plt.ylabel('Correlation Coefficient')
    plt.title('PLCC and SROCC over Epochs')
    plt.legend()
    plt.show()

