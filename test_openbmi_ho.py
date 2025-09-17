from ptfca import *
import numpy as np
import torch
import os

Fs = 250
n_length = 1000
n_ch = 20

output_addr = "outputs/OpenBMI_HO"

for subject in range(1,55):
    rec_file = rf"{output_addr}/sub{subject}_state.pt"
    if os.path.exists(rec_file):
        continue

    TX = np.load(fr"F:\dataset\MIKU\{subject}traindata.npy")  # 288 x 22 x 1000
    TY = np.load(fr"F:\dataset\MIKU\{subject}trainlabel.npy")  # 288
    EX = np.load(fr"F:\dataset\MIKU\{subject}testdata.npy")  # 288 x 22 x 1000
    EY = np.load(fr"F:\dataset\MIKU\{subject}testlabel.npy")  # 288

    select_classes = [0,1]
    train_select_ind = np.isin(TY, select_classes)
    TX = TX[train_select_ind]
    TY = TY[train_select_ind]
    test_select_ind = np.isin(EY, select_classes)
    EX = EX[test_select_ind]
    EY = EY[test_select_ind]

    TX = torch.tensor(TX).float().cuda()
    TY = torch.tensor(TY).long().cuda()
    EX = torch.tensor(EX).float().cuda()
    EY = torch.tensor(EY).long().cuda()

    # Define PTFCA
    ptfca = PTFCA(Fs, n_ch, n_length)
    # Window Optimization
    ptfca_state = ptfca.fit(TX, TY, max_windows=20, seed=0)
    # NN-Training
    nn_state = ptfca.NN_fit(ptfca_state, TX, TY, seed=0)
    TF = ptfca.NN_transform(nn_state, TX)
    EF = ptfca.NN_transform(nn_state, EX)
    # SVM-Training
    svm_state = ptfca.svm_fit(TF, TY, seed=0)
    score = ptfca.svm_score(svm_state, EF, EY)
    rec = {
        "ptfca_state": ptfca_state,
        "nn_state": nn_state,
        "svm_state": svm_state,
        "score": score,
    }
    os.makedirs(output_addr, exist_ok=True)
    torch.save(rec, rec_file)

    with open(rf"{output_addr}/score.txt", 'a') as f:
        f.write(f"Subject {subject}: {score}\n")

