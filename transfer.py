import numpy as np
import cv2
import glob
import natsort
import os
import scipy.io as scio
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Camera:
    def __init__(self, intrinsic, rotation, translation):
        self.K = intrinsic  # 内参矩阵
        self.R = rotation  # 旋转矩阵
        self.t = translation  # 平移矩阵或相机中心

    def C2W(self, point):
        # 相机坐标系到世界坐标系的转换
        temp1 = (point - self.t.T).T
        return np.dot(self.R.T, (point - self.t.T).T)

    def W2C(self, point):
        # 世界坐标系到相机坐标系的转换
        return np.dot(self.R, point) + self.t

    def C2I(self, point):
        # 相机坐标系到图像坐标系的转换
        p = np.dot(self.K, point)
        return np.array([p[0] / p[2], p[1] / p[2]])

    def W2I(self, point):
        # 世界坐标系到图像坐标系的转换
        return self.C2I(self.W2C(point))

    def I2C(self, point, depth):
        # 图像坐标系到相机坐标系的转换（使用深度）
        p = np.array([point[0], point[1], depth])
        p[0] *= p[2]
        p[1] *= p[2]

        return np.dot(np.linalg.inv(self.K), p)

    def I2W(self, point, depth):
        # 图像坐标系到世界坐标系的转换（使用深度）

        return self.C2W(self.I2C(point, depth))


def get_posit(posit, heatmap, depth):
    dist_min = 0
    dist_max = 9
    width = 9
    pose_size = 3

    dp = np.zeros(16)
    hm = np.zeros([16, 2])

    for n in range(15):
        dp[n] = (np.argmax(depth[n, :]) + 0.5) / 64 * pose_size - pose_size/2

        hm_m = np.argmax(heatmap[n, :, :])
        hm[n, :] = ((np.array([hm_m // 64, hm_m - 64 * (hm_m // 64)]) + 0.5) / 64 * pose_size -
                    pose_size/2).reshape(1, 2) * np.array([[-1, 1]])

    dp[15] = 0
    hm[15, :] = np.array([[0, 0]])

    po_m_x = np.argmax(np.max(posit, axis=1))
    po_m_z = np.argmax(np.max(posit, axis=0))
    po = (np.array([po_m_x, po_m_z]) + 0.5) / 128 * np.array([width, dist_max - dist_min]) + np.array([-8, dist_min])

    return po, hm, dp

def Cumsum_cdf(DATA):
    denominator = len(DATA)

    # #对获得的表格整体按照索引自小到大进行排序
    DATA=np.sort(DATA)
    # # 每个数据出现频数除以数据总数才能获得该数据的概率
    # #将频数转换成概率
    Fre_df=np.array(range(1,denominator+1))/denominator
    # #将列表列索引重命名
    count = 1
    count_pre = 0
    data_out = []
    for n in range(1, denominator):
        if DATA[n] == DATA[n-count] and n != denominator-1:
            Fre_df[n-count_pre-count] += 1/denominator
            count += 1
        elif DATA[n] != DATA[n-count] and n == denominator-1:
            data_out.append(DATA[n])
        else:
            data_out.append(DATA[n-1])
            if count != 1:
                Fre_df[n-count_pre-count+1:-count_pre-count] = Fre_df[n-count_pre:-count_pre-1]
                count_pre += count - 1
                count = 1

    Fre_df = Fre_df[0:-count_pre-count]
    data_out = np.array(data_out)

    return data_out, Fre_df

def main():
    batch_size = 16

    a = 0.00001819066483438419
    b = -0.0006339165332867748
    c = 0.0001516654648021437

    # 左镜头的内参，如焦距
    left_camera_matrix = np.array(
        [[1051.019951877919, 0, 680.0292238680915], [0, 1051.275121825469, 511.4991108395923], [0., 0., 1.]])
    right_camera_matrix = np.array(
        [[1050.320016099385, 0, 649.1927471496678], [0, 1050.253992675795, 521.3616965450534], [0., 0., 1.]])

    # 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
    left_distortion = np.array(
        [[-0.021310808543156, 0.148536221112070, -0.176929682420418, 0.001990017858355, 0.003534982273656]])
    right_distortion = np.array(
        [[-0.016398640448005, 0.102895600950445, -0.094286786286682, 0.002877284363279, 0.003643226094397]])

    # 旋转矩阵
    R = np.array([[0.999987607433429, 0.001139389007192, -0.004846315327732],
                  [-0.001145750978695, 0.999998485353073, -0.001310170314880],
                  [0.004844815193621, 0.001315706749037, 0.999987398261343]])
    # 平移矩阵
    T = np.array([-127.1862204754893, 0.744505070263697, -6.732168503429264])

    size = (1280, 720)

    camera_l_transfer = Camera(left_camera_matrix, R, T)

    # # # # # # # # # # # #
    # # Total Error Count #
    # # # # # # # # # # # #
    dir_depth_out = [f for f in os.listdir('Single Person Output') if f.startswith('depth_stg4_')]
    dir_depth_out = natsort.natsorted(dir_depth_out)
    dir_depth_ori = [f for f in os.listdir('Single Person Output') if f.startswith('depth_test_stg4_')]
    dir_depth_ori = natsort.natsorted(dir_depth_ori)
    dir_heatmap_out = [f for f in os.listdir('Single Person Output') if f.startswith('heatmap_stg4_')]
    dir_heatmap_out = natsort.natsorted(dir_heatmap_out)
    dir_heatmap_ori = [f for f in os.listdir('Single Person Output') if f.startswith('heatmap_test_stg4_')]
    dir_heatmap_ori = natsort.natsorted(dir_heatmap_ori)
    dir_posit_out = [f for f in os.listdir('Single Person Output') if f.startswith('posit_stg4_')]
    dir_posit_out = natsort.natsorted(dir_posit_out)
    dir_posit_ori = [f for f in os.listdir('Single Person Output') if f.startswith('posit_test_stg4_')]
    dir_posit_ori = natsort.natsorted(dir_posit_ori)
    L = 24

    err_pose = []
    err_pose_x_mae = 0
    err_pose_y_mae = 0
    err_pose_z_mae = 0
    err_pose_joint_mae = np.zeros(16)
    err_pose_mse = 0
    err_pose_x_mse = 0
    err_pose_y_mse = 0
    err_pose_z_mse = 0
    err_pose_joint_mse = np.zeros(16)
    err_posit = []
    err_posit_x_mae = 0
    err_posit_z_mae = 0
    err_posit_mse = 0
    err_posit_x_mse = 0
    err_posit_z_mse = 0
    for n in range(24):  # (0, 4608//16, 12):
        depth_out_batch = scio.loadmat('Single Person Output/' + dir_depth_out[n])['Depth']
        depth_ori_batch = scio.loadmat('Single Person Output/' + dir_depth_ori[n])['Depth']
        heatmap_out_batch = scio.loadmat('Single Person Output/' + dir_heatmap_out[n])['HeatMap']
        heatmap_ori_batch = scio.loadmat('Single Person Output/' + dir_heatmap_ori[n])['HeatMap']
        posit_out_batch = scio.loadmat('Single Person Output/' + dir_posit_out[n])['Position']
        posit_ori_batch = scio.loadmat('Single Person Output/' + dir_posit_ori[n])['Position']
        for m in range(batch_size):
            depth_out = depth_out_batch[m, :, :].reshape([15, 64])
            depth_ori = depth_ori_batch[m, :, :].reshape([15, 64])
            heatmap_out = heatmap_out_batch[m, :, :, :].reshape([15, 64, 64])
            heatmap_ori = heatmap_ori_batch[m, :, :, :].reshape([15, 64, 64])
            posit_out = posit_out_batch[m, :, :].reshape([128, 128])
            posit_ori = posit_ori_batch[m, :, :].reshape([128, 128])

            po_out, hm_out, dp_out = get_posit(posit_out, heatmap_out, depth_out)
            po_ori, hm_ori, dp_ori = get_posit(posit_ori, heatmap_ori, depth_ori)

            err_pose.append((((dp_out - dp_ori) ** 2 + np.sum((hm_out - hm_ori) ** 2, 1)) ** 0.5).mean())
            err_pose_x_mae += (np.abs(hm_out - hm_ori)[:, 1]).mean() / (L * batch_size)
            err_pose_y_mae += (np.abs(hm_out - hm_ori)[:, 0]).mean() / (L * batch_size)
            err_pose_z_mae += np.abs(dp_out - dp_ori).mean() / (L * batch_size)
            err_pose_joint_mae += (((dp_out - dp_ori) ** 2 + np.sum((hm_out - hm_ori) ** 2, 1)) ** 0.5) / (
                        L * batch_size)

            err_posit.append(((po_out - po_ori) ** 2).sum() ** 0.5)
            err_posit_x_mae += np.abs((po_out - po_ori)[0]) / (L * batch_size)
            err_posit_z_mae += np.abs((po_out - po_ori)[1]) / (L * batch_size)

            err_pose_mse += ((dp_out - dp_ori) ** 2 + np.sum((hm_out - hm_ori) ** 2, 1)).mean() / (
                    L * batch_size)
            err_pose_x_mse += ((hm_out - hm_ori)[:, 1] ** 2).mean() / (L * batch_size)
            err_pose_y_mse += ((hm_out - hm_ori)[:, 0] ** 2).mean() / (L * batch_size)
            err_pose_z_mse += ((dp_out - dp_ori) ** 2).mean() / (L * batch_size)
            err_pose_joint_mse += ((dp_out - dp_ori) ** 2 + np.sum((hm_out - hm_ori) ** 2, 1)) / (
                    L * batch_size)

            err_posit_mse += ((po_out - po_ori) ** 2).sum() / (L * batch_size)
            err_posit_x_mse += ((po_out - po_ori)[0] ** 2) / (L * batch_size)
            err_posit_z_mse += ((po_out - po_ori)[1] ** 2) / (L * batch_size)

    err_pose = np.array(err_pose)
    err_posit = np.array(err_posit)
    err_pose_mae = err_pose.mean()
    err_posit_mae = err_posit.mean()

    err_pose_out, cdf_pose = Cumsum_cdf(err_pose)
    err_posit_out, cdf_posit = Cumsum_cdf(err_posit)
    scio.savemat(
        './single_person_annotation_5/pose_err.mat',
        {'CDF': cdf_pose, 'Err': err_pose_out})
    scio.savemat(
        './single_person_annotation_5/posit_err.mat',
        {'CDF': cdf_posit, 'Err': err_posit_out})
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    ax1.plot(err_pose_out, cdf_pose)
    ax1.set_xlabel('Average Pose Estimation Error (m)')
    ax1.set_ylabel('Empirical CDF')
    fig2 = plt.figure(2)
    ax2 = fig2.add_subplot(111)
    ax2.plot(err_posit_out, cdf_posit)
    ax2.set_xlabel('Localization Error (m)')
    ax2.set_ylabel('Empirical CDF')

    print(
        f'Mean Absolute Error: \nPose: {err_pose_mae}  Pose_x: {err_pose_x_mae}  Pose_y: {err_pose_y_mae}  Pose_z: {err_pose_z_mae}')
    print(f'Pose_each_joint: \n{err_pose_joint_mae}')
    print(f'Position: {err_posit_mae}  Position_x: {err_posit_x_mae}  Position_z:{err_posit_z_mae}')

    print(
        f'\nMean Square Error: \nPose: {err_pose_mse}  Pose_x: {err_pose_x_mse}  Pose_y: {err_pose_y_mse}  Pose_z: {err_pose_z_mse}')
    print(f'Pose_each_joint: \n{err_pose_joint_mse}')
    print(f'Position: {err_posit_mse}  Position_x: {err_posit_x_mse}  Position_z:{err_posit_z_mse}')
    plt.show()

    # # # # # # # # # # #
    # # # Pose Display  #
    # # # # # # # # # # #
    # skeleton = np.array(
    #     [[14, 0], [0, 15], [15, 13], [13, 7], [13, 10], [7, 8], [8, 9], [10, 11], [11, 12], [0, 1], [1, 2], [2, 3],
    #      [0, 4], [4, 5], [5, 6]])
    #
    # for batch in range(24):
    #     root = './Single Person Output/'
    #     dir_depth_out_batch_disp = root + 'depth_stg4_' + str(batch) + '_2.mat'
    #     dir_depth_ori_batch_disp = root + 'depth_test_stg4_' + str(batch) + '_2.mat'
    #     dir_heatmap_out_batch_disp = root + 'heatmap_stg4_' + str(batch) + '_2.mat'
    #     dir_heatmap_ori_batch_disp = root + 'heatmap_test_stg4_' + str(batch) + '_2.mat'
    #     dir_posit_out_batch_disp = root + 'posit_stg4_' + str(batch) + '_2.mat'
    #     dir_posit_ori_batch_disp = root + 'posit_test_stg4_' + str(batch) + '_2.mat'
    #     dir_img_batch = root + 'img_test_stg4_' + str(batch) + '_2.mat'
    #
    #     depth_out_batch_disp = scio.loadmat(dir_depth_out_batch_disp)['Depth']
    #     depth_ori_batch_disp = scio.loadmat(dir_depth_ori_batch_disp)['Depth']
    #     heatmap_out_batch_disp = scio.loadmat(dir_heatmap_out_batch_disp)['HeatMap']
    #     heatmap_ori_batch_disp = scio.loadmat(dir_heatmap_ori_batch_disp)['HeatMap']
    #     posit_out_batch_disp = scio.loadmat(dir_posit_out_batch_disp)['Position']
    #     posit_ori_batch_disp = scio.loadmat(dir_posit_ori_batch_disp)['Position']
    #     img_batch = scio.loadmat(dir_img_batch)['img']
    #
    #     for k in range(0, batch_size):
    #         depth_out_disp = depth_out_batch_disp[k, :, :].reshape([15, 64])
    #         depth_ori_disp = depth_ori_batch_disp[k, :, :].reshape([15, 64])
    #         heatmap_out_disp = heatmap_out_batch_disp[k, :, :, :].reshape([15, 64, 64])
    #         heatmap_ori_disp = heatmap_ori_batch_disp[k, :, :, :].reshape([15, 64, 64])
    #         posit_out_disp = posit_out_batch_disp[k, :, :].reshape([128, 128])
    #         posit_ori_disp = posit_ori_batch_disp[k, :, :].reshape([128, 128])
    #
    #         po_out_disp, hm_out_disp, dp_out_disp = get_posit(posit_out_disp, heatmap_out_disp, depth_out_disp)
    #         po_ori_disp, hm_ori_disp, dp_ori_disp = get_posit(posit_ori_disp, heatmap_ori_disp, depth_ori_disp)
    #
    #         fig = plt.figure(1)
    #         ax = fig.add_subplot(111, projection='3d')
    #         fig2 = plt.figure(2)
    #         ax2 = fig2.add_subplot(111, projection='3d')
    #         fig3 = plt.figure(3)
    #         ax3 = fig3.add_subplot(111)
    #
    #         for n in range(16):
    #             ax.scatter(hm_ori_disp[n, 1], hm_ori_disp[n, 0], -dp_ori_disp[n], c='b', marker='^')
    #             ax.scatter(hm_out_disp[n, 1], hm_out_disp[n, 0], -dp_out_disp[n], c='r', marker='o')
    #
    #             ax2.scatter(-hm_ori_disp[n, 1] + po_ori_disp[0], hm_ori_disp[n, 0], dp_ori_disp[n] + po_ori_disp[1], c='b', marker='^')
    #             ax2.scatter(-hm_out_disp[n, 1] + po_out_disp[0], hm_out_disp[n, 0], dp_out_disp[n] + po_out_disp[1], c='r', marker='o')
    #
    #         for n in range(15):
    #             x_ori = np.zeros([2])
    #             y_ori = np.zeros([2])
    #             z_ori = np.zeros([2])
    #             x_ori[0] = hm_ori_disp[skeleton[n, 0], 1]
    #             y_ori[0] = hm_ori_disp[skeleton[n, 0], 0]
    #             z_ori[0] = -dp_ori_disp[skeleton[n, 0]]
    #             x_ori[1] = hm_ori_disp[skeleton[n, 1], 1]
    #             y_ori[1] = hm_ori_disp[skeleton[n, 1], 0]
    #             z_ori[1] = -dp_ori_disp[skeleton[n, 1]]
    #             ax.plot(x_ori, y_ori, z_ori, c='b')
    #
    #             x_out = np.zeros([2])
    #             y_out = np.zeros([2])
    #             z_out = np.zeros([2])
    #             x_out[0] = hm_out_disp[skeleton[n, 0], 1]
    #             y_out[0] = hm_out_disp[skeleton[n, 0], 0]
    #             z_out[0] = -dp_out_disp[skeleton[n, 0]]
    #             x_out[1] = hm_out_disp[skeleton[n, 1], 1]
    #             y_out[1] = hm_out_disp[skeleton[n, 1], 0]
    #             z_out[1] = -dp_out_disp[skeleton[n, 1]]
    #             ax.plot(x_out, y_out, z_out, c='r')
    #
    #             x_ori2 = np.zeros([2])
    #             y_ori2 = np.zeros([2])
    #             z_ori2 = np.zeros([2])
    #             x_ori2[0] = -hm_ori_disp[skeleton[n, 0], 1] + po_ori_disp[0]
    #             y_ori2[0] = hm_ori_disp[skeleton[n, 0], 0]
    #             z_ori2[0] = dp_ori_disp[skeleton[n, 0]] + po_ori_disp[1]
    #             x_ori2[1] = -hm_ori_disp[skeleton[n, 1], 1] + po_ori_disp[0]
    #             y_ori2[1] = hm_ori_disp[skeleton[n, 1], 0]
    #             z_ori2[1] = dp_ori_disp[skeleton[n, 1]] + po_ori_disp[1]
    #             ax2.plot(x_ori2, y_ori2, z_ori2, c='b')
    #
    #             x_out2 = np.zeros([2])
    #             y_out2 = np.zeros([2])
    #             z_out2 = np.zeros([2])
    #             x_out2[0] = -hm_out_disp[skeleton[n, 0], 1] + po_out_disp[0]
    #             y_out2[0] = hm_out_disp[skeleton[n, 0], 0]
    #             z_out2[0] = dp_out_disp[skeleton[n, 0]] + po_out_disp[1]
    #             x_out2[1] = -hm_out_disp[skeleton[n, 1], 1] + po_out_disp[0]
    #             y_out2[1] = hm_out_disp[skeleton[n, 1], 0]
    #             z_out2[1] = dp_out_disp[skeleton[n, 1]] + po_out_disp[1]
    #             ax2.plot(x_out2, y_out2, z_out2, c='r')
    #
    #         ax.set_xlabel('X')
    #         ax.set_ylabel('Y')
    #         ax.set_zlabel('Z')
    #         ax.view_init(elev=90, azim=0, roll=90)
    #         ax.axis('equal')
    #         ax2.set_xlabel('X')
    #         ax2.set_ylabel('Y')
    #         ax2.set_zlabel('Z')
    #         ax2.view_init(elev=90, azim=0, roll=90)
    #         ax2.axis('equal')
    #         ax3.imshow((img_batch[k, :, :, :] / 255).reshape([720, 2560, 3]))
    #         print(f'batch={batch}, k={k}')
    #         plt.show()
    #         input()
    #         plt.close('all')


if __name__ == '__main__':
    main()
