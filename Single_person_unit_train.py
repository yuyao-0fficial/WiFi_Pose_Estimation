# author: Schwarzer_land
from network_files.Single_Person_Estimation_unit import PoseNet
from utils.Single_Person_Estimator_Loss_3 import criterion

import os
import cv2
import h5py
import torch
import time
import random
import natsort
from tqdm import tqdm
import numpy as np
import scipy.io as scio
from torch import nn
import tensorflow as tf
import pandas as pd
from torch.utils.data import DataLoader
from torch import optim
# from torchsummary import summary
from Single_person_Dataset import Single_Person_Dataset


def main():
    # 设置基本参数信息
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 如果GPU不够用，可以用cpu
    # device = torch.device('cpu')
    cpu_num = 20  # 这里设置成你想运行的CPU个数
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    torch.set_num_threads(cpu_num)

    stg = 4
    continu = 1
    L_R = 0.002
    flag_train = 0  # 1是训练，0是测试
    sche = 1
    losses_weight = 0.5
    loss_posit_weight = 0.5
    max_weight = 0.99999999
    pro_weight_1 = 0.7
    pro_weight_2 = 0.3
    total = 4608
    batch_size = 256
    batch_val = 128
    batch_test = 16
    epoch = 4000

    # 创建模型
    model = PoseNet(stg=stg, flag_train=flag_train)
    root = './single_person_annotation_5/single person/'
    if stg == 0:
        params = [p for p in model.parameters() if p.requires_grad]  # 定义需要优化的参数
        nadam = optim.NAdam(params, lr=L_R, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004,
                            foreach=None)
        if sche == 0:
            scheduler = optim.lr_scheduler.OneCycleLR(nadam, max_lr=0.02, pct_start=0.025, steps_per_epoch=192,
                                                      epochs=epoch)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(nadam, factor=0.1, patience=10, verbose=True, eps=0)
        metrics_names = ['loss_train', 'loss_val']
        loss_train = []
        loss_val = []
        # 加载数据
        train_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/train_conti.mat', mode='r')['csi'][:], 0, 3), 1, 2)
        total_train = train_csi_ori.shape[0]
        p_train = list(range(total_train))
        # random.shuffle(p_train)
        p_train = np.reshape(p_train, [-1, batch_size])
        train_csi_ori = np.swapaxes(train_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
        train_csi_ori = np.append(np.append(train_csi_ori, np.zeros([total_train, 135, 6]), axis=2),
                                  np.zeros([total_train, 25, 96]), axis=1)
        train_csi = torch.from_numpy(train_csi_ori).float()
        del train_csi_ori
        train_csi = (train_csi -
                     torch.matmul(torch.min(torch.min(train_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                  torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
        train_csi = train_csi / \
                    torch.matmul(torch.max(torch.max(train_csi, 1).values, 1).values.reshape(-1, 1),
                                 torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
        val_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/val_conti.mat', mode='r')['csi'][:], 0, 3), 1, 2)
        total_val = val_csi_ori.shape[0]
        p_val = [range(total_val)]
        # random.shuffle(p_val)
        p_val = np.reshape(p_val, [-1, batch_val])
        val_csi_ori = np.swapaxes(val_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
        val_csi_ori = np.append(np.append(val_csi_ori, np.zeros([total_val, 135, 6]), axis=2),
                                np.zeros([total_val, 25, 96]), axis=1)
        val_csi = torch.from_numpy(val_csi_ori).float()
        del val_csi_ori
        val_csi = (val_csi -
                   torch.matmul(torch.min(torch.min(val_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
        val_csi = val_csi / \
                  torch.matmul(torch.max(torch.max(val_csi, 1).values, 1).values.reshape(-1, 1),
                               torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
        print('CSI data loaded!\n')
        if continu == 1 or flag_train != 1:
            # 加载过去训练数据
            weights = [f for f in os.listdir('save_weights') if
                       f.startswith('AutoEncoder_stg' + str(stg - 1 + continu))]
            weights = natsort.natsorted(weights)[-1]
            checkpoint = torch.load('save_weights/' + weights)
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print('Load ' + weights + ' Successfully!')
        model.to(device)

        name = 'save_weights/AutoEncoder_stg' + str(stg) + '_' + str(time.localtime().tm_year) + '_' + str(
            time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(
            time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(
            time.localtime().tm_sec) + '.pth'
        if flag_train == 1:
            if continu == 1:
                min_avg_loss = checkpoint['min_loss']
            else:
                min_avg_loss = 9999
            # 开始训练
            model.train()
            for e in range(epoch):
                print("\nepoch {}/{}".format(e + 1, epoch))
                progBar = tf.keras.utils.Progbar(total // batch_size, stateful_metrics=metrics_names)
                loss_temp_train = 0
                loss_temp_val = 0
                for i in range(total_train // batch_size):
                    ori_csi_train = train_csi[p_train[i, :], :, :].to(device)
                    recov_csi_train = model(ori_csi_train)
                    loss = torch.div((ori_csi_train - recov_csi_train['out']).square().sum(),
                                     ori_csi_train.square().sum())
                    loss_temp_train += loss.item()
                    values = [('loss_train', loss_temp_train / (i + 1))]
                    nadam.zero_grad()
                    loss.requires_grad_(True)
                    loss.backward()
                    nadam.step()
                    progBar.update(i, values=values)

                for j in range(total_val // batch_val):
                    ori_csi_val = val_csi[p_val[j, :], :, :].to(device)
                    recov_csi_val = model(ori_csi_val)
                    loss = torch.div((ori_csi_val - recov_csi_val['out']).square().sum(),
                                     ori_csi_val.square().sum())
                    loss_temp_val += loss.item()

                avg_loss_train = loss_temp_train / (i + 1)
                avg_loss_val = loss_temp_val / (j + 1)
                tran_val_gap = avg_loss_train - avg_loss_val
                loss_train.append(avg_loss_train)
                loss_val.append(avg_loss_val)
                values = [('loss_val', avg_loss_val), ('train_val_gap', tran_val_gap)]

                if sche == 0:
                    scheduler.step()
                else:
                    scheduler.step(avg_loss_val)
                # print(
                #     f'第{e + 1}个epoch,平均训练损失loss_train={avg_loss_train}，平均验证损失loss_val={avg_loss_val}')
                progBar.update(i + 1, values=values, finalize=True)
                if min_avg_loss > avg_loss_val:
                    min_avg_loss = avg_loss_val
                    # 保存权重
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': nadam.state_dict(),
                        'min_loss': avg_loss_val
                    }, name)
                    print('Weights saved!')

            history = {'loss_train': loss_train, 'loss_val': loss_val}
            pd.DataFrame(history).plot()

        else:
            for i in range(total_train // batch_size):
                ori_csi_train = train_csi[p_train[i, :], :, :].to(device)
                encoder_output = model(ori_csi_train)
                scio.savemat(
                    './Encoder_output/train/stg_' + str(
                        stg) + '_' + str(i) + '.mat',
                    {'Encoder_output': encoder_output['out'].cpu().detach().numpy()})

            for i in range(total_val // batch_val):
                ori_csi_val = val_csi[p_val[i, :], :, :].to(device)
                encoder_output = model(ori_csi_val)
                scio.savemat(
                    './Encoder_output/val/stg_' + str(
                        stg) + '_' + str(i) + '.mat',
                    {'Encoder_output': encoder_output['out'].cpu().detach().numpy()})

    elif stg < 3:
        params = [p for p in model.parameters() if p.requires_grad]  # 定义需要优化的参数
        nadam = optim.NAdam(params, lr=L_R, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004,
                            foreach=None)
        if sche == 0:
            scheduler = optim.lr_scheduler.OneCycleLR(nadam, max_lr=0.02, pct_start=0.025, steps_per_epoch=192,
                                                      epochs=epoch)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(nadam, factor=0.1, patience=10, verbose=True, eps=0)
        metrics_names = ['loss_train', 'loss_val']
        loss_train = []
        loss_val = []
        # 加载数据
        root = './Encoder_output/'
        encoder_out_train = [i for i in os.listdir(root + 'train') if i.startswith('stg_' + str(stg - 1))]
        code_list_train = [os.path.join(root + 'train/', i) for i in encoder_out_train]
        encoder_out_val = [i for i in os.listdir(root + 'val') if i.startswith('stg_' + str(stg - 1))]
        code_list_val = [os.path.join(root + 'val/', i) for i in encoder_out_val]
        if flag_train != 1 or continu == 1:
            # 加载过去训练数据
            weights = [f for f in os.listdir('save_weights') if
                       f.startswith('AutoEncoder_stg' + str(stg - 1 + continu))]
            weights = natsort.natsorted(weights)[-1]
            checkpoint = torch.load('save_weights/' + weights)
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print('Load ' + weights + ' Successfully!')
        model.to(device)

        name = 'save_weights/AutoEncoder_stg' + str(stg) + '_' + str(time.localtime().tm_year) + '_' + str(
            time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(
            time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(time.localtime().tm_sec) + '.pth'
        if flag_train == 1:
            if continu == 1:
                min_avg_loss = checkpoint['min_loss']
            else:
                min_avg_loss = 9999
            # 开始训练
            model.train()
            for e in range(epoch):
                print("\nepoch {}/{}".format(e + 1, epoch))
                progBar = tf.keras.utils.Progbar(total // batch_size, stateful_metrics=metrics_names)
                loss_temp_train = 0
                loss_temp_val = 0
                count_train = 0
                count_val = 0
                for i in code_list_train:
                    code = scio.loadmat(i)['Encoder_output']
                    code_train = (torch.from_numpy(code).float()).to(device)
                    recov_code_train = model(code_train)
                    loss = torch.div((code_train - recov_code_train['out']).square().sum(), code_train.square().sum())
                    loss_temp_train += loss.item()
                    values = [('loss_train', loss_temp_train / (count_train + 1))]
                    nadam.zero_grad()
                    loss.requires_grad_(True)
                    loss.backward()
                    nadam.step()
                    progBar.update(count_train, values=values)
                    count_train += 1

                for j in code_list_val:
                    code = scio.loadmat(j)['Encoder_output']
                    code_val = (torch.from_numpy(code).float()).to(device)
                    recov_code_val = model(code_val)
                    loss = torch.div((code_val - recov_code_val['out']).square().sum(), code_val.square().sum())
                    loss_temp_val += loss.item()
                    count_val += 1

                avg_loss_train = loss_temp_train / count_train
                avg_loss_val = loss_temp_val / count_val
                tran_val_gap = avg_loss_train - avg_loss_val
                loss_train.append(avg_loss_train)
                loss_val.append(avg_loss_val)
                values = [('loss_val', avg_loss_val), ('train_val_gap', tran_val_gap)]

                if sche == 0:
                    scheduler.step()
                else:
                    scheduler.step(avg_loss_val)
                # print(f'第{e+1}个epoch,平均训练损失loss_train={avg_loss_train}，平均验证损失loss_val={avg_loss_val}')
                progBar.update(count_train, values=values, finalize=True)
                if min_avg_loss > avg_loss_val:
                    min_avg_loss = avg_loss_val
                    # 保存权重
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': nadam.state_dict(),
                        'min_loss': min_avg_loss
                    }, name)
                    print('Weights saved!')

            history = {'loss_train': loss_train, 'loss_val': loss_val}
            pd.DataFrame(history).plot()

        else:
            train_count = 0
            val_count = 0
            for i in code_list_train:
                code = scio.loadmat(i)['Encoder_output']
                code_train = (torch.from_numpy(code).float()).to(device)
                recov_code_train = model(code_train)
                scio.savemat(
                    './Encoder_output/train/stg_' + str(
                        stg) + '_' + str(train_count) + '.mat',
                    {'Encoder_output': recov_code_train['out'].cpu().detach().numpy()})
                train_count += 1

            for j in code_list_val:
                code = scio.loadmat(j)['Encoder_output']
                code_val = (torch.from_numpy(code).float()).to(device)
                recov_code_val = model(code_val)
                scio.savemat(
                    './Encoder_output/val/stg_' + str(
                        stg) + '_' + str(val_count) + '.mat',
                    {'Encoder_output': recov_code_val['out'].cpu().detach().numpy()})
                val_count += 1

    elif stg == 3:
        params = [p for p in model.parameters() if p.requires_grad]  # 定义需要优化的参数
        nadam = optim.NAdam(params, lr=L_R, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004,
                            foreach=None)
        if sche == 0:
            scheduler = optim.lr_scheduler.OneCycleLR(nadam, max_lr=0.02, pct_start=0.025, steps_per_epoch=192,
                                                      epochs=epoch)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(nadam, factor=0.1, patience=10, verbose=True, eps=0)
        metrics_names = ['loss_train', 'loss_val']
        loss_train = []
        loss_val = []
        # 加载数据
        train_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/train_conti.mat', mode='r')['csi'][:], 0, 3), 1,
                                    2)
        total_train = train_csi_ori.shape[0]
        p_train = list(range(total_train))
        # random.shuffle(p_train)
        p_train = np.reshape(p_train, [-1, batch_size])
        train_csi_ori = np.swapaxes(train_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
        train_csi_ori = np.append(np.append(train_csi_ori, np.zeros([total_train, 135, 6]), axis=2),
                                  np.zeros([total_train, 25, 96]), axis=1)
        train_csi = torch.from_numpy(train_csi_ori).float()
        del train_csi_ori
        train_csi = (train_csi -
                     torch.matmul(torch.min(torch.min(train_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                  torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
        train_csi = train_csi / \
                    torch.matmul(torch.max(torch.max(train_csi, 1).values, 1).values.reshape(-1, 1),
                                 torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
        val_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/val_conti.mat', mode='r')['csi'][:], 0, 3), 1, 2)
        total_val = val_csi_ori.shape[0]
        p_val = [range(total_val)]
        # random.shuffle(p_val)
        p_val = np.reshape(p_val, [-1, batch_val])
        val_csi_ori = np.swapaxes(val_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
        val_csi_ori = np.append(np.append(val_csi_ori, np.zeros([total_val, 135, 6]), axis=2),
                                np.zeros([total_val, 25, 96]), axis=1)
        val_csi = torch.from_numpy(val_csi_ori).float()
        del val_csi_ori
        val_csi = (val_csi -
                   torch.matmul(torch.min(torch.min(val_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
        val_csi = val_csi / \
                  torch.matmul(torch.max(torch.max(val_csi, 1).values, 1).values.reshape(-1, 1),
                               torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
        print('CSI data loaded!\n')
        if continu == 1 or flag_train != 1:
            # 加载过去训练数据
            weights = [f for f in os.listdir('save_weights') if
                       f.startswith('AutoEncoder_stg' + str(stg - 1 + continu))]
            weights = natsort.natsorted(weights)[-1]
            checkpoint = torch.load('save_weights/' + weights)
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
            print('Load ' + weights + ' Successfully!')
        else:
            for n in range(3):
                # 加载过去训练数据
                weights = [f for f in os.listdir('save_weights') if
                           f.startswith('AutoEncoder_stg' + str(n))]
                weights = natsort.natsorted(weights)[-1]
                checkpoint = torch.load('save_weights/' + weights)
                model_dict = model.state_dict()
                state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict.keys()}
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
                print('Load ' + weights + ' Successfully!')
        model.to(device)

        name = 'save_weights/AutoEncoder_stg' + str(stg) + '_' + str(time.localtime().tm_year) + '_' + str(
            time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(
            time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(
            time.localtime().tm_sec) + '.pth'
        if flag_train == 1:
            if continu == 1:
                min_avg_loss = checkpoint['min_loss']
            else:
                min_avg_loss = 9999
            # 开始训练
            model.train()
            for e in range(epoch):
                print("\nepoch {}/{}".format(e + 1, epoch))
                progBar = tf.keras.utils.Progbar(total // batch_size, stateful_metrics=metrics_names)
                loss_temp_train = 0
                loss_temp_val = 0
                for i in range(total_train // batch_size):
                    ori_csi_train = train_csi[p_train[i, :], :, :].to(device)
                    recov_csi_train = model(ori_csi_train)
                    loss = torch.div((ori_csi_train - recov_csi_train['out']).square().sum(),
                                     ori_csi_train.square().sum())
                    loss_temp_train += loss.item()
                    values = [('loss_train', loss_temp_train / (i + 1))]
                    nadam.zero_grad()
                    loss.requires_grad_(True)
                    loss.backward()
                    nadam.step()
                    progBar.update(i, values=values)

                for j in range(total_val // batch_val):
                    ori_csi_val = val_csi[p_val[j, :], :, :].to(device)
                    recov_csi_val = model(ori_csi_val)
                    loss = torch.div((ori_csi_val - recov_csi_val['out']).square().sum(),
                                     ori_csi_val.square().sum())
                    loss_temp_val += loss.item()

                avg_loss_train = loss_temp_train / (i + 1)
                avg_loss_val = loss_temp_val / (j + 1)
                tran_val_gap = avg_loss_train - avg_loss_val
                loss_train.append(avg_loss_train)
                loss_val.append(avg_loss_val)
                values = [('loss_val', avg_loss_val), ('train_val_gap', tran_val_gap)]

                if sche == 0:
                    scheduler.step()
                else:
                    scheduler.step(avg_loss_val)
                # print(
                #     f'第{e + 1}个epoch,平均训练损失loss_train={avg_loss_train}，平均验证损失loss_val={avg_loss_val}')
                progBar.update(i + 1, values=values, finalize=True)
                if min_avg_loss > avg_loss_val:
                    min_avg_loss = avg_loss_val
                    # 保存权重
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': nadam.state_dict(),
                        'min_loss': avg_loss_val
                    }, name)
                    print('Weights saved!')

            history = {'loss_train': loss_train, 'loss_val': loss_val}
            pd.DataFrame(history).plot()

        else:
            for i in range(total_train // batch_size):
                ori_csi_train = train_csi[p_train[i, :], :, :].to(device)
                encoder_output = model(ori_csi_train)
                scio.savemat(
                    './Encoder_output/train/stg_' + str(
                        stg) + '_' + str(i) + '.mat',
                    {'Encoder_output': encoder_output['out'].cpu().detach().numpy()})

            for i in range(total_val // batch_val):
                ori_csi_val = val_csi[p_val[i, :], :, :].to(device)
                encoder_output = model(ori_csi_val)
                scio.savemat(
                    './Encoder_output/val/stg_' + str(
                        stg) + '_' + str(i) + '.mat',
                    {'Encoder_output': encoder_output['out'].cpu().detach().numpy()})

    elif stg == 4:
        if flag_train == 1:
            # 加载数据
            train_dphm = h5py.File(root + '/pose/train_conti.mat', mode='r')['dphm']
            val_dphm = h5py.File(root + '/pose/val_conti.mat', mode='r')['dphm']
            train_posit = torch.from_numpy(np.swapaxes(train_dphm['posit'][:], 0, 2).reshape(-1, 1, 128, 128)).float()
            val_posit = torch.from_numpy(np.swapaxes(val_dphm['posit'][:], 0, 2).reshape(-1, 1, 128, 128)).float()
            print('Position data loaded!\n')
            train_hm = torch.from_numpy(np.swapaxes(np.swapaxes(train_dphm['hm'][:], 0, 3), 1, 2)).float()
            val_hm = torch.from_numpy(np.swapaxes(np.swapaxes(val_dphm['hm'][:], 0, 3), 1, 2)).float()
            print('Heatmap data loaded!\n')
            train_depth = torch.from_numpy(np.swapaxes(train_dphm['depth'][:], 0, 2)).float()
            val_depth = torch.from_numpy(np.swapaxes(val_dphm['depth'][:], 0, 2)).float()
            print('Depth data loaded!\n')
            del train_dphm, val_dphm
            train_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/train_conti.mat', mode='r')['csi'][:], 0, 3),
                                        1, 2)
            total_train = train_csi_ori.shape[0]
            p_train = [range(total_train)]
            # random.shuffle(p_train)
            p_train = np.reshape(p_train, [-1, batch_size])
            train_csi_ori = np.swapaxes(train_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
            train_csi_ori = np.append(np.append(train_csi_ori, np.zeros([total_train, 135, 6]), axis=2),
                                      np.zeros([total_train, 25, 96]), axis=1)
            train_csi = torch.from_numpy(train_csi_ori).float()
            del train_csi_ori
            train_csi = (train_csi -
                         torch.matmul(torch.min(torch.min(train_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                      torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
            train_csi = train_csi / \
                        torch.matmul(torch.max(torch.max(train_csi, 1).values, 1).values.reshape(-1, 1),
                                     torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
            val_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/val_conti.mat', mode='r')['csi'][:], 0, 3), 1,
                                      2)
            total_val = val_csi_ori.shape[0]
            p_val = [range(total_val)]
            # random.shuffle(p_val)
            p_val = np.reshape(p_val, [-1, batch_val])
            val_csi_ori = np.swapaxes(val_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
            val_csi_ori = np.append(np.append(val_csi_ori, np.zeros([total_val, 135, 6]), axis=2),
                                    np.zeros([total_val, 25, 96]), axis=1)
            val_csi = torch.from_numpy(val_csi_ori).float()
            del val_csi_ori
            val_csi = (val_csi -
                       torch.matmul(torch.min(torch.min(val_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                    torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
            val_csi = val_csi / \
                      torch.matmul(torch.max(torch.max(val_csi, 1).values, 1).values.reshape(-1, 1),
                                   torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
            print('CSI data loaded!\n')

            model_AE = PoseNet(3)
            if continu == 1:
                model_posit = PoseNet(5)
                # 加载过去训练数据
                weights_SPE = [f for f in os.listdir('save_weights') if
                               f.startswith('Single_Person_Estimator_' + str(stg))]
                weights_SPE = natsort.natsorted(weights_SPE)[-1]
                checkpoint_SPE = torch.load('save_weights/' + weights_SPE)
                model_dict = model.state_dict()
                state_dict_SPE = {k: v for k, v in checkpoint_SPE['model_state_dict'].items() if k in model_dict.keys()}
                model_dict.update(state_dict_SPE)
                model.load_state_dict(model_dict)
                print('Load ' + weights_SPE + ' Successfully!')

                # weights_posit = [f for f in os.listdir('save_weights') if
                #                  f.startswith('Single_Person_Estimator_' + str(5))]
                # weights_posit = natsort.natsorted(weights_posit)[-1]
                # checkpoint_posit = torch.load('save_weights/' + weights_posit)
                # model_dict = model.state_dict()
                # state_dict_posit = {k: v for k, v in checkpoint_posit['model_state_dict'].items() if
                #                     k in model_dict.keys()}
                # # model_dict.update(state_dict_posit)
                # # model.load_state_dict(model_dict)
                # model_dict_posit = model_posit.state_dict()
                # model_dict_posit.update(state_dict_posit)
                # for k, v in model.named_parameters():
                #     if k in model_dict_posit.keys():
                #         v.requires_grad = False
                # print('Load ' + weights_posit + ' Successfully!')
                #
                # weights_AE = [f for f in os.listdir('save_weights') if f.startswith('AutoEncoder_stg' + str(3))]
                # weights_AE = natsort.natsorted(weights_AE)[-1]
                # checkpoint_AE = torch.load('save_weights/' + weights_AE)
                # model_dict = model.state_dict()
                # state_dict_AE = {k: v for k, v in checkpoint_AE['model_state_dict'].items() if k in model_dict.keys()}
                # model_dict_AE = model_AE.state_dict()
                # model_dict_AE.update(state_dict_AE)
                # for k, v in model.named_parameters():
                #     if k in model_dict_AE.keys():
                #         v.requires_grad = False
                # print('Load ' + weights_AE + ' Successfully!')

            else:
                weights_AE = [f for f in os.listdir('save_weights') if f.startswith('AutoEncoder_stg' + str(3))]
                weights_AE = natsort.natsorted(weights_AE)[-1]
                checkpoint_AE = torch.load('save_weights/' + weights_AE)
                model_dict = model.state_dict()
                state_dict_AE = {k: v for k, v in checkpoint_AE['model_state_dict'].items() if k in model_dict.keys()}
                model_dict.update(state_dict_AE)
                model.load_state_dict(model_dict)
                model_dict_AE = model_AE.state_dict()
                model_dict_AE.update(state_dict_AE)
                for k, v in model.named_parameters():
                    if k in model_dict_AE.keys():
                        v.requires_grad = False
                print('Load ' + weights_AE + ' Successfully!')

            # params = [p for p in model.parameters() if p.requires_grad]  # 定义需要优化的参数
            params = filter(lambda p: p.requires_grad, model.parameters())
            # 定义优化器
            nadam = optim.NAdam(params, lr=L_R, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004,
                                foreach=None)
            # scheduler = optim.lr_scheduler.ExponentialLR(nadam, 0.9986)
            # scheduler = optim.lr_scheduler.CosineAnnealingLR(nadam, T_max=120, eta_min=0)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(nadam, factor=0.1, patience=100, verbose=True, eps=0)
            for name, value in model.named_parameters():
                print(name, value.requires_grad)  # 打印所有参数requires_grad属性，True或False
            model.to(device)

            name = 'save_weights/Single_Person_Estimator_' + str(stg) + '_' + str(
                time.localtime().tm_year) + '_' + str(
                time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(
                time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(
                time.localtime().tm_sec) + '.pth'
            name_hist = 'Single Person Output/history_' + str(stg) + '_' + str(
                time.localtime().tm_year) + '_' + str(
                time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(
                time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(
                time.localtime().tm_sec) + '.mat'
            if continu == 1:
                min_avg_loss = 9999  # checkpoint_SPE['min_loss']
                min_avg_loss_2 = 9999  # checkpoint_SPE['min_avg_loss']
            else:
                min_avg_loss = 9999
                min_avg_loss_2 = 9999  # checkpoint_SPE['min_avg_loss']

            metrics_names = ['loss_train', 'loss_train_0', 'loss_train_4', 'loss_train_8', 'loss_hm_train',
                             'loss_dp_train',
                             'loss_po_train', 'loss_val', 'loss_hm_val', 'loss_dp_val', 'loss_po_val']

            # 开始训练
            loss_train = []
            loss_hm_train = []
            loss_dp_train = []
            loss_po_train = []
            loss_diff_train = []
            loss_main_train = []
            loss_val = []
            loss_hm_val = []
            loss_dp_val = []
            loss_po_val = []
            loss_diff_val = []
            loss_main_val = []
            model.train()
            for e in range(epoch):
                print("\nepoch {}/{}".format(e + 1, epoch))
                progBar = tf.keras.utils.Progbar(total_train // batch_size, stateful_metrics=metrics_names)
                loss = {}
                loss['losses'] = {}
                loss_temp_train = 0
                loss_hm_temp_train = 0
                loss_dp_temp_train = 0
                loss_po_temp_train = 0
                loss_dff_temp_train = 0
                loss_main_temp_train = 0
                for i in range(total_train // batch_size):
                    csi_train = train_csi[p_train[i, :], :, :].to(device)
                    model_out = model(csi_train)

                    heatmap_train = train_hm[p_train[i, :], :, :, :].to(device)
                    depth_train = train_depth[p_train[i, :], :, :].to(device)
                    posit_train = train_posit[p_train[i, :], :, :].to(device)
                    loss = criterion(model_out, heatmap_train, depth_train, posit_train, stg=stg, phase=0,
                                       loss_hmdp_weight=losses_weight, device=device,
                                       loss_posit_weight=loss_posit_weight,
                                       max_weight=max_weight)
                    del depth_train, heatmap_train, posit_train

                    loss_temp_train += loss['main'].item()
                    loss_hm_temp_train += loss['losses']['loss_hm'].item()
                    loss_dp_temp_train += loss['losses']['loss_dp'].item()
                    loss_po_temp_train += loss['losses']['loss_po'].item()
                    loss_dff_temp_train += loss['losses']['diff'].item()
                    loss_main_temp_train += loss['losses']['main'].item()
                    values = [('loss_train', loss_temp_train / (i + 1)),
                              ('loss_hm_train', loss_hm_temp_train / (i + 1)),
                              ('loss_dp_train', loss_dp_temp_train / (i + 1)),
                              ('loss_po_train', loss_po_temp_train / (i + 1)),
                              ('loss_diff_train', loss_dff_temp_train / (i + 1)),
                              ('loss_main_train', loss_main_temp_train / (i + 1))]

                    nadam.zero_grad()
                    loss['main'].requires_grad_(True)
                    loss['main'].backward()

                    nadam.step()
                    progBar.update(i, values=values)
                    torch.cuda.empty_cache()

                loss_temp_val = 0
                loss_hm_temp_val = 0
                loss_dp_temp_val = 0
                loss_po_temp_val = 0
                loss_dff_temp_val = 0
                loss_main_temp_val = 0
                for j in range(total_val // batch_val):
                    # print(j)
                    csi_val = val_csi[p_val[j, :], :, :].to(device)
                    model_out = model(csi_val)

                    heatmap_val = val_hm[p_val[j, :], :, :, :].to(device)
                    depth_val = val_depth[p_val[j, :], :, :].to(device)
                    posit_val = val_posit[p_val[j, :], :, :, :].to(device)
                    loss = criterion(model_out, heatmap_val, depth_val, posit_val, stg=stg, phase=0,
                                       loss_hmdp_weight=losses_weight, device=device,
                                       loss_posit_weight=loss_posit_weight,
                                       max_weight=max_weight)
                    del depth_val, heatmap_val, posit_val

                    loss_temp_val += loss['main'].item()
                    loss_hm_temp_val += loss['losses']['loss_hm'].item()
                    loss_dp_temp_val += loss['losses']['loss_dp'].item()
                    loss_po_temp_val += loss['losses']['loss_po'].item()
                    loss_dff_temp_val += loss['losses']['diff'].item()
                    loss_main_temp_val += loss['losses']['main'].item()
                    torch.cuda.empty_cache()

                avg_loss_train = loss_temp_train / (i + 1)
                avg_loss_val = loss_temp_val / (j + 1)
                avg_loss_hm_train = loss_hm_temp_train / (i + 1)
                avg_loss_hm_val = loss_hm_temp_val / (j + 1)
                avg_loss_dp_train = loss_dp_temp_train / (i + 1)
                avg_loss_dp_val = loss_dp_temp_val / (j + 1)
                avg_loss_po_train = loss_po_temp_train / (i + 1)
                avg_loss_po_val = loss_po_temp_val / (j + 1)
                avg_loss_dff_train = loss_dff_temp_train / (i + 1)
                avg_loss_dff_val = loss_dff_temp_val / (j + 1)
                avg_loss_main_train = loss_main_temp_train / (i + 1)
                avg_loss_main_val = loss_main_temp_val / (j + 1)

                loss_train.append(avg_loss_train)
                loss_val.append(avg_loss_val)
                loss_hm_train.append(avg_loss_hm_train)
                loss_hm_val.append(avg_loss_hm_val)
                loss_dp_train.append(avg_loss_dp_train)
                loss_dp_val.append(avg_loss_dp_val)
                loss_po_train.append(avg_loss_po_train)
                loss_po_val.append(avg_loss_po_val)
                loss_diff_train.append(avg_loss_dff_train)
                loss_diff_val.append(avg_loss_dff_val)
                loss_main_train.append(avg_loss_main_train)
                loss_main_val.append(avg_loss_main_val)

                values = [('loss_val', avg_loss_val),
                          ('loss_hm_val', avg_loss_hm_val),
                          ('loss_dp_val', avg_loss_dp_val),
                          ('loss_po_val', avg_loss_po_val),
                          ('loss_diff_val', avg_loss_dff_val)]

                progBar.update(i + 1, values=values, finalize=True)

                # scheduler.step()
                scheduler.step(avg_loss_val)
                if min_avg_loss > avg_loss_val:
                    min_avg_loss = avg_loss_val
                    # 保存权重
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': nadam.state_dict(),
                        'min_loss': min_avg_loss,
                        'min_avg_loss': min_avg_loss_2
                    }, name)
                    print('Weights saved!')
                elif min_avg_loss_2 > avg_loss_main_val:
                    min_avg_loss_2 = avg_loss_main_val
                    # 保存权重
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': nadam.state_dict(),
                        'min_loss': min_avg_loss,
                        'min_avg_loss': min_avg_loss_2
                    }, name)
                    print('Weights saved!')

                history = {'loss_train': loss_train, 'loss_hm_train': loss_hm_train, 'loss_dp_train': loss_dp_train,
                           'loss_diff_train': loss_diff_train, 'loss_main_train': loss_main_train, 'loss_val': loss_val,
                           'loss_hm_val': loss_hm_val, 'loss_dp_val': loss_dp_val, 'loss_diff_val': loss_diff_val,
                           'loss_main_val': loss_main_val}
                scio.savemat(name_hist, {'history': history})

        else:
            # 加载数据
            test_dphm = h5py.File(root + '/pose/test_conti.mat', mode='r')['dphm']
            test_posit = torch.from_numpy(np.swapaxes(test_dphm['posit'][:], 0, 2).reshape(-1, 1, 128, 128)).float()
            print('\nPosition data loaded!\n')
            test_hm = torch.from_numpy(np.swapaxes(np.swapaxes(test_dphm['hm'][:], 0, 3), 1, 2)).float()
            print('Heatmap data loaded!\n')
            test_depth = torch.from_numpy(np.swapaxes(test_dphm['depth'][:], 0, 2)).float()
            print('Depth data loaded!\n')
            del test_dphm
            test_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/test_conti.mat', mode='r')['csi'][:], 0, 3),
                                       1, 2)
            total_test = test_csi_ori.shape[0]
            p_test = list(range(total_test))
            # random.shuffle(p_test)
            p_test = np.reshape(p_test, [-1, batch_test])
            test_csi_ori = np.swapaxes(test_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
            test_csi_ori = np.append(np.append(test_csi_ori, np.zeros([total_test, 135, 6]), axis=2),
                                     np.zeros([total_test, 25, 96]), axis=1)
            test_csi = torch.from_numpy(test_csi_ori).float()
            del test_csi_ori
            test_csi = (test_csi -
                        torch.matmul(torch.min(torch.min(test_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                     torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
            test_csi = test_csi / \
                       torch.matmul(torch.max(torch.max(test_csi, 1).values, 1).values.reshape(-1, 1),
                                    torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
            print('CSI data loaded!\n')
            test_jpg = [f for f in os.listdir('single_person_annotation_5/single person/jpeg/test') if
                        f.endswith('.jpg')]
            test_jpg = natsort.natsorted(test_jpg)

            weights_SPE = [f for f in os.listdir('save_weights') if
                           f.startswith('Single_Person_Estimator_' + str(stg))]
            weights_SPE = natsort.natsorted(weights_SPE)[-1]
            checkpoint_SPE = torch.load('save_weights/' + weights_SPE)
            model_dict = model.state_dict()
            state_dict_SPE = {k: v for k, v in checkpoint_SPE['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict_SPE)
            model.load_state_dict(model_dict)
            print('Load ' + weights_SPE + ' Successfully!\n')

            # model_posit = PoseNet(5)
            # weights_posit = [f for f in os.listdir('save_weights') if
            #                  f.startswith('Single_Person_Estimator_' + str(stg))]
            # weights_posit = natsort.natsorted(weights_posit)[-1]
            # checkpoint_posit = torch.load('save_weights/' + weights_posit)
            # model_dict = model.state_dict()
            # model_posit_dict = model_posit.state_dict()
            # state_dict_posit = {k: v for k, v in checkpoint_posit['model_state_dict'].items() if
            #                     k in model_posit_dict.keys()}
            # model_dict.update(state_dict_posit)
            # model.load_state_dict(model_dict)
            # print('Load ' + weights_posit + ' Successfully!')

            model.to(device)

            # metrics_names = ['loss_test', 'loss_hm_test', 'loss_dp_test']

            loss_temp_test = 0
            loss_hm_temp_test = 0
            loss_dp_temp_test = 0
            loss_po_temp_test = 0
            loss_diff_temp_test = 0
            loss_main_temp_test = 0
            jpeg_test = np.zeros([batch_test, 720, 2560, 3])
            t_temp = 0
            for i in range(0, total_test // batch_test):
                csi_test = test_csi[p_test[i, :], :, :].to(device)
                posit_test = test_posit[p_test[i, :], :, :, :].to(device)
                heatmap_test = test_hm[p_test[i, :], :, :, :].to(device)
                depth_test = test_depth[p_test[i, :], :, :].to(device)
                for j in range(batch_test):
                    jpeg_test[j, :, :, :] = cv2.imread(
                        os.path.join('single_person_annotation_5/single person/jpeg/test',
                                     test_jpg[p_test[i, j]])).reshape(1, 720, 2560, 3)
                t_s = time.time()
                model_out = model(csi_test)
                t_e = time.time()
                scio.savemat(
                    './Single Person Output/posit_stg' + str(
                        stg) + '_' + str(i) + '_2.mat',
                    {'Position': model_out['Position'].cpu().detach().numpy()})
                scio.savemat(
                    './Single Person Output/heatmap_stg' + str(
                        stg) + '_' + str(i) + '_2.mat',
                    {'HeatMap': model_out['HeatMap'].cpu().detach().numpy()})
                scio.savemat(
                    './Single Person Output/depth_stg' + str(
                        stg) + '_' + str(i) + '_2.mat',
                    {'Depth': model_out['Depth'].cpu().detach().numpy()})
                scio.savemat(
                    './Single Person Output/posit_test_stg' + str(
                        stg) + '_' + str(i) + '_2.mat',
                    {'Position': posit_test.cpu().detach().numpy()})
                scio.savemat(
                    './Single Person Output/heatmap_test_stg' + str(
                        stg) + '_' + str(i) + '_2.mat',
                    {'HeatMap': heatmap_test.cpu().detach().numpy()})
                scio.savemat(
                    './Single Person Output/depth_test_stg' + str(
                        stg) + '_' + str(i) + '_2.mat',
                    {'Depth': depth_test.cpu().detach().numpy()})
                scio.savemat(
                    './Single Person Output/img_test_stg' + str(
                        stg) + '_' + str(i) + '_2.mat',
                    {'img': jpeg_test})
                loss = criterion(model_out, heatmap_test, depth_test, posit_test, stg=stg, phase=0,
                                 loss_hmdp_weight=losses_weight, device=device, loss_posit_weight=loss_posit_weight,
                                 max_weight=max_weight)
                t_temp = t_temp + t_e - t_s
                loss_temp_test += loss['main'].item()
                loss_hm_temp_test += loss['losses']['loss_hm'].item()
                loss_dp_temp_test += loss['losses']['loss_dp'].item()
                loss_po_temp_test += loss['losses']['loss_po'].item()
                loss_diff_temp_test += loss['losses']['diff'].item()
                loss_main_temp_test += loss['losses']['main'].item()
                torch.cuda.empty_cache()

                print(f'Batch {i+1} finished!\n')

            avg_loss_test = loss_temp_test / (i + 1)
            avg_t = t_temp / (i + 1)
            print(f'平均测试损失loss_test={avg_loss_test}，平均每batch用时t={avg_t}')
            avg_loss_hm_test = loss_hm_temp_test / (i + 1)
            avg_loss_dp_test = loss_dp_temp_test / (i + 1)
            avg_loss_po_test = loss_po_temp_test / (i + 1)
            avg_loss_diff_test = loss_diff_temp_test / (i + 1)
            avg_loss_main_test = loss_main_temp_test / (i + 1)
            print(
                f'loss_hm_test={avg_loss_hm_test}，loss_dp_test={avg_loss_dp_test}, loss_po_test={avg_loss_po_test}, loss_diff_test={avg_loss_diff_test}, loss_main_test={avg_loss_main_test}')

    else:
        # 加载数据
        train_dphm = h5py.File(root + '/pose/train_conti.mat', mode='r')['dphm']
        val_dphm = h5py.File(root + '/pose/val_conti.mat', mode='r')['dphm']
        train_posit = torch.from_numpy(np.swapaxes(train_dphm['posit'][:], 0, 2).reshape(-1, 1, 128, 128)).float()
        val_posit = torch.from_numpy(np.swapaxes(val_dphm['posit'][:], 0, 2).reshape(-1, 1, 128, 128)).float()
        print('Position data loaded!\n')
        del train_dphm, val_dphm
        train_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/train_conti.mat', mode='r')['csi'][:], 0, 3),
                                    1, 2)
        total_train = train_csi_ori.shape[0]
        p_train = [range(total_train)]
        # random.shuffle(p_train)
        p_train = np.reshape(p_train, [-1, batch_size])
        train_csi_ori = np.swapaxes(train_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
        train_csi_ori = np.append(np.append(train_csi_ori, np.zeros([total_train, 135, 6]), axis=2),
                                  np.zeros([total_train, 25, 96]), axis=1)
        train_csi = torch.from_numpy(train_csi_ori).float()
        del train_csi_ori
        train_csi = (train_csi -
                     torch.matmul(torch.min(torch.min(train_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                  torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
        train_csi = train_csi / \
                    torch.matmul(torch.max(torch.max(train_csi, 1).values, 1).values.reshape(-1, 1),
                                 torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
        val_csi_ori = np.swapaxes(np.swapaxes(h5py.File(root + '/csi/val_conti.mat', mode='r')['csi'][:], 0, 3), 1,
                                  2)
        total_val = val_csi_ori.shape[0]
        p_val = [range(total_val)]
        # random.shuffle(p_val)
        p_val = np.reshape(p_val, [-1, batch_val])
        val_csi_ori = np.swapaxes(val_csi_ori.reshape(-1, 27, 90, 5), 2, 3).reshape(-1, 135, 90)
        val_csi_ori = np.append(np.append(val_csi_ori, np.zeros([total_val, 135, 6]), axis=2),
                                np.zeros([total_val, 25, 96]), axis=1)
        val_csi = torch.from_numpy(val_csi_ori).float()
        del val_csi_ori
        val_csi = (val_csi -
                   torch.matmul(torch.min(torch.min(val_csi, 1).values, 1).values.reshape(-1, 1).float(),
                                torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96]))
        val_csi = val_csi / \
                  torch.matmul(torch.max(torch.max(val_csi, 1).values, 1).values.reshape(-1, 1),
                               torch.from_numpy(np.ones([1, 160 * 96])).float()).reshape([-1, 160, 96])
        print('CSI data loaded!\n')

        if continu == 1:
            model_posit = PoseNet(5)
            # 加载过去训练数据
            weights_SPE = [f for f in os.listdir('save_weights') if
                           f.startswith('Single_Person_Estimator_' + str(stg))]
            weights_SPE = natsort.natsorted(weights_SPE)[-1]
            checkpoint_SPE = torch.load('save_weights/' + weights_SPE)
            model_dict = model.state_dict()
            state_dict_SPE = {k: v for k, v in checkpoint_SPE['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict_SPE)
            model.load_state_dict(model_dict)
            print('Load ' + weights_SPE + ' Successfully!')

            # weights_posit = [f for f in os.listdir('save_weights') if
            #                f.startswith('Single_Person_Estimator_' + str(5))]
            # weights_posit = natsort.natsorted(weights_posit)[-1]
            # checkpoint_posit = torch.load('save_weights/' + weights_posit)
            # model_dict = model.state_dict()
            # state_dict_posit = {k: v for k, v in checkpoint_posit['model_state_dict'].items() if k in model_dict.keys()}
            # model_dict.update(state_dict_posit)
            # model.load_state_dict(model_dict)
            # model_dict_posit = model_posit.state_dict()
            # model_dict_posit.update(state_dict_posit)
            # for k, v in model.named_parameters():
            #     if k in model_dict_posit.keys():
            #         v.requires_grad = False
            # print('Load ' + weights_posit + ' Successfully!')

            model_AE = PoseNet(3)
            weights_AE = [f for f in os.listdir('save_weights') if f.startswith('AutoEncoder_stg' + str(3))]
            weights_AE = natsort.natsorted(weights_AE)[-1]
            checkpoint_AE = torch.load('save_weights/' + weights_AE)
            model_dict = model.state_dict()
            state_dict_AE = {k: v for k, v in checkpoint_AE['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict_AE)
            model_dict_AE = model_AE.state_dict()
            model_dict_AE.update(state_dict_AE)
            for k, v in model.named_parameters():
                if k in model_dict_AE.keys():
                    v.requires_grad = False
            print('Load ' + weights_AE + ' Successfully!')
        else:
            model_AE = PoseNet(3)
            weights_AE = [f for f in os.listdir('save_weights') if f.startswith('AutoEncoder_stg' + str(3))]
            weights_AE = natsort.natsorted(weights_AE)[-1]
            checkpoint_AE = torch.load('save_weights/' + weights_AE)
            model_dict = model.state_dict()
            state_dict_AE = {k: v for k, v in checkpoint_AE['model_state_dict'].items() if k in model_dict.keys()}
            model_dict.update(state_dict_AE)
            model.load_state_dict(model_dict)
            model_dict_AE = model_AE.state_dict()
            model_dict_AE.update(state_dict_AE)
            for k, v in model.named_parameters():
                if k in model_dict_AE.keys():
                    v.requires_grad = False
            print('Load ' + weights_AE + ' Successfully!')

        # params = [p for p in model.parameters() if p.requires_grad]  # 定义需要优化的参数
        params = filter(lambda p: p.requires_grad, model.parameters())
        # 定义优化器
        nadam = optim.NAdam(params, lr=L_R, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004,
                            foreach=None)
        # scheduler = optim.lr_scheduler.ExponentialLR(nadam, 0.9986)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(nadam, T_max=120, eta_min=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(nadam, factor=0.1, patience=100, verbose=True, eps=0)
        for name, value in model.named_parameters():
            print(name, value.requires_grad)  # 打印所有参数requires_grad属性，True或False
        model.to(device)

        name = 'save_weights/Single_Person_Estimator_' + str(stg) + '_' + str(
            time.localtime().tm_year) + '_' + str(
            time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(
            time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(
            time.localtime().tm_sec) + '.pth'
        name_hist = 'Single Person Output/history_' + str(stg) + '_' + str(
            time.localtime().tm_year) + '_' + str(
            time.localtime().tm_mon) + '_' + str(time.localtime().tm_mday) + '_' + str(
            time.localtime().tm_hour) + '_' + str(time.localtime().tm_min) + '_' + str(
            time.localtime().tm_sec) + '.mat'
        if continu == 1:
            min_avg_loss = 9999  # checkpoint_SPE['min_loss']
            min_avg_loss_2 = 9999  # checkpoint_SPE['min_avg_loss']
        else:
            min_avg_loss = 9999
            min_avg_loss_2 = 9999  # checkpoint_SPE['min_avg_loss']

        metrics_names = ['loss_train', 'loss_val']

        # 开始训练
        loss_train = []
        loss_val = []
        heatmap_train = torch.tensor(0).to(device)
        depth_train = torch.tensor(0).to(device)
        heatmap_val = torch.tensor(0).to(device)
        depth_val = torch.tensor(0).to(device)
        model.train()
        for e in range(epoch):
            print("\nepoch {}/{}".format(e + 1, epoch))
            progBar = tf.keras.utils.Progbar(total_train // batch_size, stateful_metrics=metrics_names)
            loss = {}
            loss['losses'] = {}
            loss_temp_train = 0
            for i in range(total_train // batch_size):
                csi_train = train_csi[p_train[i, :], :, :].to(device)
                model_out = model(csi_train)

                posit_train = train_posit[p_train[i, :], :, :].to(device)
                loss = criterion(model_out, heatmap_train, depth_train, posit_train, stg=2, phase=0,
                                 loss_hmdp_weight=losses_weight, device=device,
                                 loss_posit_weight=loss_posit_weight,
                                 max_weight=max_weight)
                del posit_train

                loss_temp_train += loss['main'].item()
                values = [('loss_train', loss_temp_train / (i + 1))]

                nadam.zero_grad()
                loss['main'].requires_grad_(True)
                loss['main'].backward()

                nadam.step()
                progBar.update(i, values=values)
                torch.cuda.empty_cache()

            loss_temp_val = 0
            for j in range(total_val // batch_val):
                # print(j)
                csi_val = val_csi[p_val[j, :], :, :].to(device)
                model_out = model(csi_val)

                posit_val = val_posit[p_val[j, :], :, :, :].to(device)
                loss = criterion(model_out, heatmap_val, depth_val, posit_val, stg=2, phase=0,
                                 loss_hmdp_weight=losses_weight, device=device,
                                 loss_posit_weight=loss_posit_weight,
                                 max_weight=max_weight)
                del posit_val

                loss_temp_val += loss['main'].item()
                torch.cuda.empty_cache()

            avg_loss_train = loss_temp_train / (i + 1)
            avg_loss_val = loss_temp_val / (j + 1)

            loss_train.append(avg_loss_train)
            loss_val.append(avg_loss_val)

            values = [('loss_val', avg_loss_val)]

            progBar.update(i + 1, values=values, finalize=True)

            # scheduler.step()
            scheduler.step(avg_loss_val)
            if min_avg_loss > avg_loss_val and (avg_loss_train - avg_loss_val)/avg_loss_val > -0.2:
                min_avg_loss = avg_loss_val
                # 保存权重
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': nadam.state_dict(),
                    'min_loss': min_avg_loss,
                    'min_avg_loss': min_avg_loss_2
                }, name)
                print('Weights saved!')

            history = {'loss_train': loss_train, 'loss_val': loss_val}
            scio.savemat(name_hist, {'history': history})


if __name__ == '__main__':
    main()
