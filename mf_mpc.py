import read_file_mfmpc as rf
import numpy as np
import math
import time


def _get_uu_mpc_iru(u_id):
    ir_u = train_im_set[u_id]

    num_ir_u = np.zeros(5, dtype=np.float)
    sqrt_num_ir_u = np.zeros(5, dtype=np.float)
    for ii in range(5):
        num_ir_u[ii] = len(ir_u[ii+1])
        sqrt_num_ir_u[ii] = math.sqrt(num_ir_u[ii])

    sum_mri = np.zeros((5, d), dtype=float)

    for ii in range(5):
        for kk in ir_u[ii+1]:
            sum_mri[ii] = sum_mri[ii] + Mr_ik[ii][kk]
        if sqrt_num_ir_u[ii] == 0:
            sum_mri[ii] = 0
            continue
        sum_mri[ii] = sum_mri[ii] / sqrt_num_ir_u[ii]

    uu_mpc = np.zeros(d, np.float)
    for ii in range(5):
        uu_mpc = uu_mpc + sum_mri[ii]
    return uu_mpc, sqrt_num_ir_u


def _test_compute():
    rmse = 0.0
    mae = 0.0
    for _record in test_set:
        u_id = _record[0]
        i_id = _record[1]
        r_ui = _record[2]
        uu_mpc = UuMpc[uid]
        p_rui = global_avg + bu[u_id] + bi[i_id] + float(np.dot(U[u_id], V[i_id])) + float(np.dot(uu_mpc, V[i_id]))

        rmse += math.pow(r_ui - p_rui, 2)
        mae += math.fabs(r_ui - p_rui)

    rmse = math.sqrt(rmse / test_r_num)
    mae = mae / test_r_num
    return rmse, mae


# train_set, yui, train_exp_set, train_im_set, test_set
data = rf.read__train__test()
train_set = data[0]
yui = data[1]
train_exp_set = data[2]
train_im_set = data[3]
test_set = data[4]

n = 943
m = 1682
train_r_num = 80000
test_r_num = 20000

global_avg = 0.0
numerator = 0.0
denominator = 0.0
for u in range(n):
    for i in range(m):
        numerator += train_set[u][i] * yui[u][i]
        denominator += yui[u][i]

global_avg = numerator / denominator

# bu bi
bu = np.zeros(n, dtype=np.float)
bi = np.zeros(m, dtype=np.float)
for u in range(n):
    numerator = 0.0
    denominator = 0.0
    for i in range(m):
        numerator += yui[u][i] * (train_set[u][i] - global_avg)
        denominator += yui[u][i]
    if numerator == 0 or denominator == 0:
        bu[u] = 0
        continue
    bu[u] = numerator / denominator

for i in range(m):
    numerator = 0.0
    denominator = 0.0
    for u in range(n):
        numerator += yui[u][i] * (train_set[u][i] - global_avg)
        denominator += yui[u][i]
    if numerator == 0 or denominator == 0:
        bi[i] = 0
        continue
    bi[i] = numerator / denominator

lr = 0.01
d = 20
T = 50

lamb = 0.01

U = np.zeros((n, d), dtype=np.float)
for u in range(n):
    for j in range(d):
        r = np.random.rand(1)
        U[u][j] = (r - 0.5) * 0.01

V = np.zeros((m, d), dtype=np.float)
'''
for i in range(m):
    for j in range(d):
        r = np.random.rand(1)
        V[i][j] = (r - 0.5) * 0.01
'''

UuMpc = np.zeros((n, d), dtype=np.float)


Mr_ik = np.zeros((5, m, d), dtype=np.float)
for k in range(5):
    for i in range(m):
        for j in range(d):
            r = np.random.rand(1)
            V[i][j] = (r - 0.5) * 0.01
            r = np.random.rand(1)
            Mr_ik[k][i][j] = (r - 0.5) * 0.01

pre_rmse = 10000
pre_mae = 10000

best_model_parm = []

whole_time_start = time.time()

for t1 in range(T):
    Rmse = 0.0
    Mae = 0.0
    iter_time_start = time.time()
    for t2 in range(train_r_num):
        index = np.random.randint(train_r_num)
        record = train_exp_set[index]
        uid = record[0]
        iid = record[1]
        rui = record[2]

        res = _get_uu_mpc_iru(uid)
        UuMpc[uid] = res[0]
        Sqrt_num_ir_u = res[1]

        pred_rui = global_avg + bu[uid] + bi[iid] + float(np.dot(U[uid], V[iid])) + float(np.dot(UuMpc[uid], V[iid]))

        if pred_rui < 1:
            pred_rui = 1
        if pred_rui > 5:
            pred_rui = 5

        eui = rui - pred_rui
        neg_eui = -eui

        temp = U[uid][:]
        tempV = np.zeros((m, d), dtype=np.float)
        for i in range(m):
            tempV[i] = V[i][:]

        U[uid] -= lr * (neg_eui * V[iid] + lamb * U[uid])
        V[iid] -= lr * (neg_eui * (temp + UuMpc[uid]) + lamb * V[iid])

        global_avg -= lr * neg_eui
        bu[uid] -= lr * (neg_eui + lamb * bu[uid])
        bi[iid] -= lr * (neg_eui + lamb * bi[iid])

        for i in range(5):
            item_list = train_im_set[uid][i+1]
            Mr_ik[i][item_list] -= lr * ((neg_eui * tempV[item_list])/Sqrt_num_ir_u[i] + lamb * Mr_ik[i][item_list])

    iter_time_end = time.time()
    print('当前迭代轮:', t1 + 1, end='  ')
    print('该轮耗时:', round(iter_time_end - iter_time_start, 4), 's')

    res = _test_compute()
    Rmse = res[0]
    Mae = res[1]
    print('RMSE:', round(Rmse, 4), end='  ')
    print('MAE:', round(Mae, 4))

    if pre_mae > Mae and pre_rmse > Rmse:
        pre_mae = Mae
        pre_rmse = Rmse
        best_model_parm = [global_avg, bu, bi, U, V, Mr_ik]

    print('全程中最小RMSE:', round(pre_rmse, 4), end='  ')
    print('最小MAE:', round(pre_mae, 4))
    lr = lr * 0.9

whole_time_end = time.time()
print('全程耗时:', (whole_time_end - whole_time_start), 's')
