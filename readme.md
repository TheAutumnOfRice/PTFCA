This is the code for the paper `Principal Time–Frequency Component Analysis for Interpretable Motor Imagery BCIs`

The original results and visualization outputs are stored in `/outputs`.

The code of the core models `ptfca.py` will be made public as soon as the paper is accepted.

## Multi-class decoding

`ptfca.py` supports multi-class problems through two strategies, selected with `mc_mode`:

| `mc_mode` | Binary tasks fitted | Task key in the returned state |
| --- | --- | --- |
| `"ovo"` (default) | one per class pair, `k*(k-1)/2` | `(c0, c1)` |
| `"ovr"` | one per class vs. the rest, `k` | `("ovr", c)` |

`class_list` selects (and re-labels) the classes to fit on, so a binary run is just a
special case. The per-task features are concatenated and decoded by one multi-class
linear SVM:

```python
from ptfca import PTFCA

ptfca = PTFCA(Fs, n_ch, n_length)

# 1) Window optimization: one binary task per class pair (or per class with mc_mode="ovr")
ptfca_state = ptfca.fit(TX, TY, max_windows=30, seed=0, mc_mode="ovo", class_list=[0, 1, 2, 3])

# 2) Multi-class feature extraction (PTFCA-Net log-variance, or the CSP-LDA alternative)
nn_state = ptfca.NN_fit(ptfca_state, TX, TY, seed=0, class_list=[0, 1, 2, 3], mc_mode="ovo")
TF = ptfca.NN_transform(nn_state, TX)
EF = ptfca.NN_transform(nn_state, EX)

# 3) Multi-class SVM on the concatenated per-task features
#    svm_mc_mode is the SVM's own decoding strategy ("ovr" / "crammer_singer"), independent
#    of the mc_mode task decomposition above.
svm_state = ptfca.svm_fit(TF, TY, seed=0, svm_mc_mode="ovr")
score = ptfca.svm_score(svm_state, EF, EY)
```

`test_iv2a_mc_ho.py` runs this end to end for the 4-class IV-2a data set:

```
python test_iv2a_mc_ho.py --subjects 1 2 3 --mc-mode ovo ovr
python test_iv2a_mc_ho.py --subjects 1 --model csp        # CSP-LDA features instead of PTFCA-Net
```

The default `fit(..., mc_mode="ovo", class_list=None)` reproduces the previous binary
behaviour.

