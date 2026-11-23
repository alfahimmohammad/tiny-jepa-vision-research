import os, pandas as pd, numpy as np
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
R = os.path.join(REPO, "results", "report_runs")   # notebook runs: result.json, train_log.csv, eval_curve.csv
e=pd.read_csv(f"{R}/h1_masking__baseline/eval_curve.csv")
cols=["step","heldout_loss","heldout_loss_shuffled_ctx","heldout_loss_mean_target","eff_rank","feat_std_mean","val_acc","acc_clean","acc_noise0.15","acc_occ40","top1_sv_frac","mean_pairwise_cos","cos_noise0.15"]
s=e[e.step.isin([0,50,100,150,200,250,300,400,500,700,1000,1500,2000,3000,5000,7500,10000,12500,15000])][cols]
print(s.round(3).to_string(index=False))
t=pd.read_csv(f"{R}/h1_masking__baseline/train_log.csv")
print("corr ctx_len vs loss (step>2000):", np.corrcoef(t[t.step>2000].ctx_len, t[t.step>2000].loss)[0,1].round(3), "ctx_len range", t.ctx_len.min(), t.ctx_len.max())
r=t.loss.rolling(100).mean(); print("min rolling train loss step", int(t.step[r.idxmin()]), round(r.min(),4), "final", round(r.iloc[-1],4))
i=e.heldout_loss.idxmin(); print("min heldout", e.step[i], round(e.heldout_loss[i],4), "val_acc", e.val_acc[i], "rank", e.eff_rank[i])
i=e["acc_noise0.15"].idxmax(); print("max noise.15", e.step[i], e["acc_noise0.15"][i])
from scipy.stats import spearmanr
ee=e[e.step>=500]; print("spearman heldout vs val_acc (step>=500):", round(spearmanr(ee.heldout_loss, ee.val_acc)[0],3), "| eff_rank vs val_acc:", round(spearmanr(ee.eff_rank, ee.val_acc)[0],3), "| eff_rank vs noise.15:", round(spearmanr(ee.eff_rank, ee['acc_noise0.15'])[0],3))
for run in ["h1_masking__scattered","h1_masking__1_large","h1_masking__2_large_overlap"]:
    e2=pd.read_csv(f"{R}/{run}/eval_curve.csv"); f=e2.iloc[-1]
    print(run, "final hl/shuf/mean", round(f.heldout_loss,3), round(f.heldout_loss_shuffled_ctx,3), round(f.heldout_loss_mean_target,3), "max noise.15", e2["acc_noise0.15"].max(), "at", int(e2.step[e2["acc_noise0.15"].idxmax()]))
