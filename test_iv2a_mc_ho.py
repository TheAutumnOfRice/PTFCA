"""Multi-class PTFCA example: 4-class IV-2a cross-session decoding.

Runs the full multi-class pipeline for every subject of the BCI Competition IV-2a
dataset (session 1 -> train, session 2 -> test) and reports the accuracy for both
multi-class strategies made available by PTFCA:

    mc_mode="ovo"  : one-vs-one, one binary PTFCA task per class pair  (k*(k-1)/2 tasks)
    mc_mode="ovr"  : one-vs-rest, one binary PTFCA task per class      (k tasks)

Both modes feed the SAME multi-class head: the per-task features are concatenated
(``CSP_LDA_transform`` / ``NN_transform``) and a multi-class linear SVM
(``svm_fit(..., svm_mc_mode=...)``) is trained on top.

Usage:
    python test_iv2a_mc_ho.py
    python test_iv2a_mc_ho.py --subjects 1 2 --classes 0 1 2 3 --mc-mode ovo ovr
    python test_iv2a_mc_ho.py --model nn --max-windows 20 --max-epoch 500
"""
import argparse
import os

import numpy as np
import torch

from ptfca import PTFCA

DATA_DIR = r"F:\dataset\BCICIV2AO"
OUTPUT_DIR = "outputs/IV2A_HO_MC"


def parse_args():
    parser = argparse.ArgumentParser(description="4-class IV-2a hold-out evaluation of PTFCA.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--classes", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--mc-mode", nargs="+", default=["ovo"], choices=["ovo", "ovr"],
                        help="PTFCA binary task decomposition: ovo (class pairs) or ovr (one-vs-rest).")
    parser.add_argument("--model", default="nn", choices=["nn", "csp"], help="nn: PTFCA-Net features, csp: CSP-LDA features.")
    parser.add_argument("--max-windows", type=int, default=30, help="Number of windows kept per binary task.")
    parser.add_argument("--max-epoch", type=int, default=2000, help="PTFCA-Net training epochs (model=nn).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--svm-c", type=float, default=0.0002)
    parser.add_argument("--svm-pca", type=int, default=128)
    parser.add_argument("--svm-mc-mode", default="ovr", choices=["ovr", "crammer_singer"],
                        help="Decoding strategy of the multi-class SVM (sklearn LinearSVC multi_class).")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_subject(data_dir, subject, classes):
    TX = np.load(os.path.join(data_dir, f"{subject}traindata.npy"))  # n_trial x n_ch x n_p
    TY = np.load(os.path.join(data_dir, f"{subject}trainlabel.npy"))  # n_trial
    EX = np.load(os.path.join(data_dir, f"{subject}testdata.npy"))
    EY = np.load(os.path.join(data_dir, f"{subject}testlabel.npy"))

    train_mask = np.isin(TY, classes)
    test_mask = np.isin(EY, classes)
    return TX[train_mask], TY[train_mask], EX[test_mask], EY[test_mask]


def evaluate_subject(args, subject, mc_mode, device):
    classes = list(args.classes)
    TX, TY, EX, EY = load_subject(args.data_dir, subject, classes)

    TX = torch.tensor(TX).float().to(device)
    TY = torch.tensor(TY).long().to(device)
    EX = torch.tensor(EX).float().to(device)
    EY = torch.tensor(EY).long().to(device)

    ptfca = PTFCA(250, TX.shape[1], TX.shape[2])
    # 1) Window optimization: one binary task per class pair (ovo) or per class (ovr).
    search_state = ptfca.fit(
        TX, TY,
        max_windows=args.max_windows,
        seed=args.seed,
        verbose=args.verbose,
        mc_mode=mc_mode,
        class_list=classes,
    )
    # 2) Multi-class feature extraction.
    if args.model == "nn":
        feature_state = ptfca.NN_fit(
            search_state, TX, TY,
            seed=args.seed,
            class_list=classes,
            mc_mode=mc_mode,
            max_epoch=args.max_epoch,
            verbose=args.verbose,
        )
        TF = ptfca.NN_transform(feature_state, TX, target="logvar")
        EF = ptfca.NN_transform(feature_state, EX, target="logvar")
    else:
        feature_state = ptfca.CSP_LDA_fit(search_state, TX, TY, class_list=classes, mc_mode=mc_mode)
        TF = ptfca.CSP_LDA_transform(feature_state, TX, target="als")
        EF = ptfca.CSP_LDA_transform(feature_state, EX, target="als")

    # 3) Multi-class SVM on the concatenated per-task features.
    #    Note: svm_mc_mode is the *SVM's* own decoding strategy ("ovr" / "crammer_singer"),
    #    independent of PTFCA's mc_mode task decomposition above.
    svm_state = ptfca.svm_fit(TF, TY, C=args.svm_c, svm_mc_mode=args.svm_mc_mode,
                              seed=args.seed, pca=args.svm_pca)
    score = ptfca.svm_score(svm_state, EF, EY)
    return score, search_state, feature_state, svm_state


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for mc_mode in args.mc_mode:
        scores = []
        for subject in args.subjects:
            print(f"[{mc_mode}] subject {subject} ...", flush=True)
            score, search_state, feature_state, _ = evaluate_subject(args, subject, mc_mode, device)
            scores.append(score)
            print(f"[{mc_mode}] subject {subject}: {score:.4f}", flush=True)
            if args.model == "csp":
                continue
            rec_file = os.path.join(args.output_dir, f"sub{subject}_{mc_mode}_state.pt")
            torch.save({
                "search_state": search_state,
                "feature_state": feature_state,
                "score": score,
                "mc_mode": mc_mode,
                "classes": list(args.classes),
            }, rec_file)

        mean_score = float(np.mean(scores))
        print(f"[{mc_mode}] mean over {len(scores)} subjects: {mean_score:.4f}", flush=True)
        rows.append((mc_mode, mean_score, scores))

    with open(os.path.join(args.output_dir, "score.txt"), "a") as f:
        for mc_mode, mean_score, scores in rows:
            f.write(f"model={args.model} mc_mode={mc_mode} classes={list(args.classes)} "
                    f"max_windows={args.max_windows} mean={mean_score:.4f}\n")
            for subject, score in zip(args.subjects, scores):
                f.write(f"  subject {subject}: {score:.4f}\n")


if __name__ == "__main__":
    main()
