import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from ptfca import *

exp_addr = "outputs/OpenBMI_HO"
for select_subject in range(1,55):
    # select_subject = 1
    target_cp = f"{exp_addr}/sub{select_subject}_state.pt"

    rec = torch.load(target_cp)
    ptfca_state = rec['ptfca_state'][(0,1)]

    # %% Activation Map
    weight = [state['afr'] for state in ptfca_state]
    mask = get_tf_region_masks(ptfca_state, 0, 4, 4, 40, 40, 40, weight).mean(dim=0)
    normed_mask = mask/mask.amax()
    plt.figure()
    ax = plt.gca()
    plt.imshow(mask, cmap="jet", vmin=0, vmax=1,origin="lower")
    ax.set_xticks([0, 10, 20, 30, 39])
    ax.set_xticklabels(['Cue', '', '', '', 'End'])
    ax.set_yticks([0, 9, 22, 36])
    ax.set_yticklabels([4, 12, 24, 36])
    plt.title("Activation Map")
    plt.savefig(f"{exp_addr}/sub{select_subject}_activation_map.png")
    # plt.show()

    # %% Select Windows
    plt.figure()
    TX = np.load(fr"F:\dataset\MIKU\{select_subject}traindata.npy")  # 288 x 22 x 1000
    TY = np.load(fr"F:\dataset\MIKU\{select_subject}trainlabel.npy")  # 288
    EX = np.load(fr"F:\dataset\MIKU\{select_subject}testdata.npy")  # 288 x 22 x 1000
    EY = np.load(fr"F:\dataset\MIKU\{select_subject}testlabel.npy")  # 288
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
    ptfca = PTFCA(250, 22, 1000)
    ptfca_state = rec['ptfca_state'][(0,1)]
    nn_state = rec['nn_state']
    TF = ptfca.NN_transform(nn_state, TX, target="logvar")
    EF = ptfca.NN_transform(nn_state, EX, target="logvar")
    TL = ptfca.NN_transform(nn_state, TX, target="logit")
    EL = ptfca.NN_transform(nn_state, EX, target="logit")

    svm_state = ptfca.svm_fit(TF, TY, win=5, seed=0)

    fusion_clf = ptfca.svm_predict(svm_state, EF, win=5)
    single_clf = torch.stack([LL.argmax(dim=-1) for LL in EL[:5]],dim=0).cpu().numpy()
    total_clf = np.concatenate([single_clf, fusion_clf[None]])
    accs = (total_clf == EY.cpu().numpy()).mean(axis=-1)
    afrs = [s['afr'] for s in ptfca_state[:5]]

    ecolors = [
        "#D9445E",
        "#F98300",
        "#5BD954",
        "#54D9D5",
        "#486BD9",
    ]
    plot_tf_box_with_afr(ptfca_state[:5], 0, 4, 4, 40,
                         linewidths=[3] * 5,
                         alphas=[1]*5,
                         ecolors=ecolors, ax=plt.gca())

    labels = [f'AFR: {afrs[ii]:.2f}\nAcc: {accs[ii] * 100:.2f}%' for ii in range(5)]
    legend_handles = [
        Patch(facecolor='none', edgecolor=color, linewidth=3, label=label)
        for color, label in zip(ecolors, labels)
    ]
    plt.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),  # Adjust for placement
        frameon=True
    )
    plt.title(fr"IV2a Subject {select_subject}: {accs[-1] * 100:.2f}%")
    plt.yticks([4., 12., 24., 36.])


    plt.gcf().supxlabel("Time (s)", y=0.05)
    plt.gcf().supylabel("Frequency (Hz)",x=0.01)

    plt.tight_layout()
    plt.savefig(f"{exp_addr}/sub{select_subject}_five_windows.png")
    # plt.show()

