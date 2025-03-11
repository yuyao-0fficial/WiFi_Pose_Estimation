# author: baiCai
import torch
from torch import nn
import numpy as np


# loss`
def criterion(inputs, heatmap, depth, posit, stg, phase, device, loss_posit_weight=0.5, loss_hmdp_weight=0.5,
              max_weight=None):
    losses = {}
    dp = 'Depth'
    if phase == 0:
        hm = 'HeatMap'
        po = 'Position'
    elif phase == 8:
        hm = 'HeatMap_8'
        po = 'Position_8'
    else:
        hm = 'HeatMap_32'
        po = 'Position_32'
    if stg == 0:
        if max_weight is None:
            loss = torch.div((depth - inputs[dp]).square().sum(1).sum(1).sum(), depth.square().sum(1).sum(1).sum())
        else:
            loss_max_up = (torch.argmax(depth, 2) - torch.argmax(inputs[dp], 2)).square().sum(1).sum().float()
            loss_max_dwn = torch.argmax(depth, 2).square().sum(1).sum().float()
            loss_pix_up = (depth - inputs[dp]).square().sum(1).sum(1).sum()
            loss_pix_dwn = depth.square().sum(1).sum(1).sum()
            if loss_max_dwn == 0:
                loss_max_dwn = 1

            loss = torch.mul(torch.tensor(max_weight).to(device),
                             torch.div(loss_max_up, loss_max_dwn)) + \
                   torch.mul(torch.tensor(1 - max_weight).to(device),
                             torch.div(loss_pix_up, loss_pix_dwn))
            if loss.equal(torch.tensor(float('inf')).to(device)):
                print(f'\n 最大值：{loss_max_up}/{loss_max_dwn}，逐像素{loss_pix_up}/{loss_pix_dwn}')

    elif stg == 1:
        if max_weight is None:
            loss = torch.div((heatmap - inputs[hm]).square().sum(1).sum(1).sum(1).sum(),
                             heatmap.square().sum(1).sum(1).sum(1).sum())
        else:
            loss_max_up = ((torch.argmax(torch.max(heatmap, 3).values, 2) -
                            torch.argmax(torch.max(inputs[hm], 3).values, 2)).square() +
                           (torch.argmax(torch.max(heatmap, 2).values, 2) -
                            torch.argmax(torch.max(inputs[hm], 2).values, 2)).square()).sum(1).sum().float()
            loss_max_dwn = (torch.argmax(torch.max(heatmap, 3).values, 2).square() +
                            torch.argmax(torch.max(heatmap, 2).values, 2).square()).sum(1).sum().float()
            loss_pix_up = (heatmap - inputs[hm]).square().sum(1).sum(1).sum(1).sum()
            loss_pix_dwn = heatmap.square().sum(1).sum(1).sum(1).sum()
            if loss_max_dwn == 0:
                loss_max_dwn = 1
            if loss_pix_dwn == 0:
                loss_pix_dwn = 0.000001

            loss = torch.mul(torch.tensor(max_weight).to(device),
                             torch.div(loss_max_up, loss_max_dwn)) + \
                   torch.mul(torch.tensor(1 - max_weight).to(device),
                             torch.div(loss_pix_up, loss_pix_dwn))

    elif stg == 2:
        if max_weight is None:
            loss = torch.div((posit - inputs[po]).square().sum(2).sum(2).sum(),
                             posit.square().sum(2).sum(2).sum())
        else:
            loss_max_up = ((torch.argmax(torch.max(posit, 3).values, 2) -
                            torch.argmax(torch.max(inputs[po], 3).values, 2)).square() +
                           (torch.argmax(torch.max(posit, 2).values, 2) -
                            torch.argmax(torch.max(inputs[po], 2).values, 2)).square()).sum().float()
            loss_max_dwn = (torch.argmax(torch.max(posit, 3).values, 2).square() +
                            torch.argmax(torch.max(posit, 2).values, 2).square()).sum().float()
            loss_pix_up = (posit - inputs[po]).square().sum(2).sum(2).sum()
            loss_pix_dwn = posit.square().sum(2).sum(2).sum()
            if loss_max_dwn == 0:
                loss_max_dwn = 1
            if loss_pix_dwn == 0:
                loss_pix_dwn = 0.000001

            loss = torch.mul(torch.tensor(max_weight).to(device),
                             torch.div(loss_max_up, loss_max_dwn)) + \
                   torch.mul(torch.tensor(1 - max_weight).to(device),
                             torch.div(loss_pix_up, loss_pix_dwn))

    else:
        if max_weight is None:
            losses['loss_po'] = torch.div((posit - inputs[po]).square().sum(2).sum(2).sum(),
                                          posit.square().sum(2).sum(2).sum())
            losses['loss_hm'] = torch.div((heatmap - inputs[hm]).square().sum(1).sum(1).sum(1).sum(),
                                          heatmap.square().sum(1).sum(1).sum(1).sum())
            losses['loss_dp'] = torch.div((depth - inputs[dp]).square().sum(1).sum(1).sum(),
                                          depth.square().sum(1).sum(1).sum())
            loss = loss_hmdp_weight * losses['loss_hm'] + (1 - loss_hmdp_weight) * losses['loss_dp']
        else:
            losses['loss_po'] = torch.mul(torch.tensor(max_weight).to(device),
                                          torch.div(((torch.argmax(torch.max(posit, 3).values, 2) -
                                                      torch.argmax(torch.max(inputs[po],
                                                                             3).values, 2)).square() +
                                                     (torch.argmax(torch.max(posit, 2).values, 2) -
                                                      torch.argmax(torch.max(inputs[po],
                                                                             2).values, 2)).square()).sum().float(),
                                                    (torch.argmax(torch.max(posit, 3).values, 2).square() +
                                                     torch.argmax(torch.max(posit, 2).values, 2).square()
                                                     ).sum().float())) + \
                                torch.mul(torch.tensor(1 - max_weight).to(device),
                                          torch.div((posit - inputs[po]).square().sum(2).sum(2).sum(),
                                                    posit.square().sum(2).sum(2).sum()))
            losses['loss_hm'] = torch.mul(torch.tensor(max_weight).to(device),
                                          torch.div((((torch.argmax(torch.max(heatmap, 3).values, 2) -
                                                       torch.argmax(torch.max(inputs[hm],
                                                                              3).values, 2)).square() +
                                                      (torch.argmax(torch.max(heatmap, 2).values, 2) -
                                                       torch.argmax(torch.max(inputs[hm],
                                                                              2).values, 2)).square())).sum(1).sum(),
                                                    (torch.argmax(torch.max(heatmap, 3).values, 2).square() +
                                                     torch.argmax(torch.max(heatmap, 2).values, 2).square()
                                                     ).sum(1).sum())) + \
                                torch.mul(torch.tensor(1 - max_weight).to(device),
                                          torch.div((heatmap - inputs[hm]).square().sum(1).sum(1).sum(1).sum(),
                                                    heatmap.square().sum(1).sum(1).sum(1).sum()))
            losses['loss_dp'] = torch.mul(torch.tensor(max_weight).to(device),
                                          torch.div((torch.argmax(depth, 2) -
                                                     torch.argmax(inputs[dp], 2)).square().sum(1).sum(),
                                                    torch.argmax(depth, 2).square().sum(1).sum())) + \
                                torch.mul(torch.tensor(1 - max_weight).to(device),
                                          torch.div((depth - inputs[dp]).square().sum(1).sum(1).sum(),
                                                    depth.square().sum(1).sum(1).sum()))

            loss_pomax_up = ((torch.argmax(torch.max(posit, 3).values, 2) -
                              torch.argmax(torch.max(inputs[po], 3).values, 2)).square() +
                             (torch.argmax(torch.max(posit, 2).values, 2) -
                              torch.argmax(torch.max(inputs[po], 2).values, 2)).square()).sum().float()
            loss_pomax_dwn = (torch.argmax(torch.max(posit, 3).values, 2).square() +
                              torch.argmax(torch.max(posit, 2).values, 2).square()).sum().float()
            loss_max_up = ((torch.argmax(torch.max(heatmap, 3).values, 2) -
                            torch.argmax(torch.max(inputs[hm], 3).values, 2)).square() +
                           (torch.argmax(torch.max(heatmap, 2).values, 2) -
                            torch.argmax(torch.max(inputs[hm], 2).values, 2)).square() +
                           (torch.argmax(depth, 2) - torch.argmax(inputs[dp], 2)).square()).sum(1).sum().float()
            loss_max_dwn = (torch.argmax(torch.max(heatmap, 3).values, 2).square() +
                            torch.argmax(torch.max(heatmap, 2).values, 2).square() +
                            torch.argmax(depth, 2).square()).sum(1).sum().float()
            loss_popix_up = (posit - inputs[po]).square().sum(2).sum(2).sum()
            loss_popix_dwn = posit.square().sum(2).sum(2).sum()
            loss_hmpix_up = (heatmap - inputs[hm]).square().sum(1).sum(1).sum(1).sum()
            loss_hmpix_dwn = heatmap.square().sum(1).sum(1).sum(1).sum()
            loss_dppix_up = (depth - inputs[dp]).square().sum(1).sum(1).sum()
            loss_dppix_dwn = depth.square().sum(1).sum(1).sum()

            if loss_pomax_dwn == 0:
                loss_pomax_dwn = 1
            if loss_max_dwn == 0:
                loss_max_dwn = 1
            if loss_popix_dwn == 0:
                loss_popix_dwn = 0.000001
            if loss_hmpix_dwn == 0:
                loss_hmpix_dwn = 0.000001
            if loss_dppix_dwn == 0:
                loss_dppix_dwn = 0.000001

            losses['main'] = torch.mul(torch.tensor(loss_posit_weight).to(device),
                                       torch.mul(torch.tensor(max_weight).to(device),
                                                 torch.div(loss_pomax_up, loss_pomax_dwn)) +
                                       torch.mul(torch.tensor(1 - max_weight).to(device),
                                                 torch.div(loss_popix_up, loss_popix_dwn))) + \
                             torch.mul(torch.tensor(1 - loss_posit_weight).to(device),
                                       torch.mul(torch.tensor(max_weight).to(device),
                                                 torch.div(loss_max_up, loss_max_dwn)) +
                                       torch.mul(torch.tensor(1 - max_weight).to(device),
                                                 torch.mul(torch.tensor(loss_hmdp_weight).to(device),
                                                           torch.div(loss_hmpix_up, loss_hmpix_dwn)) +
                                                 torch.mul(torch.tensor(1 - loss_hmdp_weight).to(device),
                                                           torch.div(loss_dppix_up, loss_dppix_dwn))))

            loss_max_posit_up = 0
            loss_max_posit_dwn = 0
            loss_max_pose_up = 0
            loss_max_pose_dwn = 0
            for k in range(1, 26):
                a = (torch.argmax(torch.max(posit, 3).values, 2)[k:-1] -
                     torch.argmax(torch.max(posit, 3).values, 2)[0:-1 - k])
                a_in = (torch.argmax(torch.max(inputs[po], 3).values, 2)[k:-1] -
                        torch.argmax(torch.max(inputs[po], 3).values, 2)[0:-1 - k])
                b = (torch.argmax(torch.max(posit, 2).values, 2)[k:-1] -
                     torch.argmax(torch.max(posit, 2).values, 2)[0:-1 - k])
                b_in = (torch.argmax(torch.max(inputs[po], 2).values, 2)[k:-1] -
                        torch.argmax(torch.max(inputs[po], 2).values, 2)[0:-1 - k])
                c = (torch.argmax(torch.max(heatmap, 3).values, 2)[k:-1, :] -
                     torch.argmax(torch.max(heatmap, 3).values, 2)[0:-1 - k, :])
                c_in = (torch.argmax(torch.max(inputs[hm], 3).values, 2)[k:-1, :] -
                        torch.argmax(torch.max(inputs[hm], 3).values, 2)[0:-1 - k, :])
                d = (torch.argmax(torch.max(heatmap, 2).values, 2)[k:-1, :] -
                     torch.argmax(torch.max(heatmap, 2).values, 2)[0:-1 - k, :])
                d_in = (torch.argmax(torch.max(inputs[hm], 2).values, 2)[k:-1, :] -
                        torch.argmax(torch.max(inputs[hm], 2).values, 2)[0:-1 - k, :])
                e = (torch.argmax(depth, 2)[k:-1, :] -
                     torch.argmax(depth, 2)[0:-1 - k, :])
                e_in = (torch.argmax(inputs[dp], 2)[k:-1, :] -
                        torch.argmax(inputs[dp], 2)[0:-1 - k, :])

                loss_max_posit_up += ((1 / torch.where(a == 0, 0.00000001, a) -
                                       1 / torch.where(a_in == 0, 0.00000001, a_in)).square() +
                                      (1 / torch.where(b == 0, 0.00000001, b) -
                                       1 / torch.where(b_in == 0, 0.00000001, b_in)).square()).sum().float() / 25

                loss_max_posit_dwn += ((1 / torch.where(a == 0, 0.00000001, a)).square() +
                                       (1 / torch.where(b == 0, 0.00000001, b)).square()).sum().float() / 25

                loss_max_pose_up += ((1 / torch.where(c == 0, 0.00000001, c) -
                                      1 / torch.where(c_in == 0, 0.00000001, c_in)).square() +
                                     (1 / torch.where(d == 0, 0.00000001, d) -
                                      1 / torch.where(d_in == 0, 0.00000001, d_in)).square() +
                                     (1 / torch.where(e == 0, 0.00000001, e) -
                                      1 / torch.where(e_in == 0, 0.00000001, e_in)).square()).sum(1).sum().float() / 25

                loss_max_pose_dwn += ((1 / torch.where(c == 0, 0.00000001, c)).square() +
                                      (1 / torch.where(d == 0, 0.00000001, d)).square() +
                                      (1 / torch.where(e == 0, 0.00000001, e)).square()).sum(1).sum().float() / 25

            if loss_max_posit_dwn == 0:
                loss_max_posit_dwn = 1
            if loss_max_pose_dwn == 0:
                loss_max_pose_dwn = 1

            losses['diff'] = torch.mul(torch.tensor(loss_posit_weight).to(device),
                                       torch.div(loss_max_posit_up, loss_max_posit_dwn)) + \
                             torch.mul(torch.tensor(1 - loss_posit_weight).to(device),
                                       torch.div(loss_max_pose_up, loss_max_pose_dwn))

            loss = torch.mul(torch.tensor(0.999).to(device), losses['main']) + \
                   torch.mul(torch.tensor(0.001).to(device), losses['diff'])

    return {'main': loss, 'losses': losses}
