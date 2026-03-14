import os
import pickle
import time
import yaml
import shutil
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from module.TCN import TemporalConvNet
from module.combiner import WeightedFeatureCombiner, MLP
from module.HGNNP import HGNNP
from utils import plot_loss, save_best

with open("./configs/train.yaml", 'r') as f:
        config = DotMap(yaml.safe_load(f))

# --------- Set seed and device ---------
torch.manual_seed(2021)
device = torch.device('cuda')

# --------- Load data ---------
with open(config.dataset_path, 'rb') as f:
    dataset: dict = pickle.load(f)

X = dataset['X'].to(device) # (|V|, T)
Y = dataset['Y'].to(device) # (|V|)

H_industry = dataset['H_industry'].to(device) # (|V|, |E_1|)
H_city = dataset['H_city'].to(device) # (|v|, |E_2|)

train_mask = dataset['train_mask'] # (|V|)
val_mask = dataset['val_mask'] # (|V|)

# --------- Build models ---------
# TCN
tcn = TemporalConvNet(1, [16, 32, 64, 64], kernel_size=3, dropout=0.2).to(device)

# HCNN+
d_in = tcn.num_channels[-1]
net_ind = HGNNP(d_in, 4*d_in, d_in, use_bn=False, drop_rate=0.0).to(device)
net_city = HGNNP(d_in, 4*d_in, d_in, use_bn=False, drop_rate=0.0).to(device)

# Combiner
combiner = WeightedFeatureCombiner()

# MLP
mlp = MLP(d_in, 8, 1).to(device)

# Optimizer & lr_scheduler
epochs = config.epochs
params = list(tcn.parameters()) + list(net_ind.parameters()) + list(net_city.parameters()) + list(combiner.parameters()) + list(mlp.parameters())
optimizer = AdamW(
    params, 
    lr=config.lr,
    weight_decay=config.weight_decay,
)
lr_scheduler = CosineAnnealingLR(
    optimizer=optimizer, 
    T_max=epochs
)

# --------- Training process ---------
def train(
        ablation: bool,
        tcn: TemporalConvNet, 
        net_ind: HGNNP, 
        net_city: HGNNP, 
        combiner: WeightedFeatureCombiner, 
        mlp: MLP, 
        num_epochs: int, 
        print_every: int = 100, 
        gen_chart: bool = False,
    ):

    save_dir = f"./exp/normal/" if not ablation else f"./exp/ablation/"
    os.makedirs(save_dir, exist_ok=True)

    if ablation:
        print('Ablation experiment!')
        net_ind = nn.Identity()
        net_city = nn.Identity()
    else:
        print('Normal experiment!')

    # create checkpoints saving directory
    os.makedirs(os.path.join(save_dir, "checkpoints/"), exist_ok=True)

    X_input = X.unsqueeze(1) # (|V|, 1, T)
    ground_truth = Y # (|V|)

    min_val_loss = None
    best_epoch = -1
    total_time = 0
    total_loss = 0

    train_loss_list = []
    val_loss_list = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tcn.train()
        net_ind.train()
        net_city.train()
        combiner.train()
        mlp.train()

        optimizer.zero_grad()

        f = tcn(X_input) # (|V|, 64)

        gf1 = f if ablation else net_ind(f, H_industry) # (|V|, 64)
        gf2 = f if ablation else net_city(f, H_city) # (|V|, 64)

        f = combiner(f, gf1, gf2) # (|V|, 64)

        f = mlp(f).squeeze(-1) # (|V|)

        # update optimizer & lr_scheduler
        loss = F.l1_loss(f[train_mask], ground_truth[train_mask])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # validation
        curr_mae, curr_mse, pred = evaluate(X, Y, val_mask, tcn, net_ind, net_city, combiner, mlp)
        
        # update records
        train_loss_list.append(loss)
        val_loss_list.append(curr_mae)
        total_loss += loss
        total_time += time.time() - t0

        # save the best model on the validation set
        if min_val_loss is None or curr_mae < min_val_loss:
            save_best(epoch, tcn, net_ind, net_city, combiner, mlp, save_dir)
            min_val_loss = curr_mae
            best_epoch = epoch
        
        # print information
        if epoch % print_every == 0:
            print(f"[Epoch {epoch:03d}] Loss: {total_loss / print_every:.4f} | Time: {total_time / print_every:.2f}s | min_mae: {min_val_loss:.4f} | best_epoch: {best_epoch}")
            total_time = 0
            total_loss = 0

    print(f'min_mae:{min_val_loss:.4f}, epoch:{best_epoch}')

    # plot the loss curve
    if gen_chart:
        MAX = config.chart.max_value
        MIN = config.chart.min_value
        if not ablation:
            plot_loss(train_loss_list, val_loss_list, 1, save_path=os.path.join(save_dir, "normal_loss.png"), set_max_val=MAX, set_min_val=MIN)
        else:
            plot_loss(train_loss_list, val_loss_list, 1, save_path=os.path.join(save_dir, "ablation_loss.png"), set_max_val=MAX, set_min_val=MIN)
    
    s = f"{min_val_loss:.4f}".split('.')
    dst = os.path.join(save_dir, f"{s[0]}_{s[1]}")
    os.makedirs(dst, exist_ok=True)

    for name in list(os.listdir(save_dir)):
        src = os.path.join(save_dir, name)
        shutil.move(src, dst)
  

@torch.no_grad()
def evaluate(
    X: torch.Tensor,
    Y: torch.Tensor,
    val_mask: torch.Tensor,
    tcn: TemporalConvNet,
    net_ind: HGNNP, 
    net_city: HGNNP,
    combiner: WeightedFeatureCombiner, 
    mlp: MLP
):
    tcn.eval()
    net_ind.eval()
    net_city.eval()
    combiner.eval()
    mlp.eval()

    X_input = X.unsqueeze(1) # (|V|, 1, 30)

    f = tcn(X_input) # (|V|, 64)
    gf1 = f if isinstance(net_ind, nn.Identity) else net_ind(f, H_industry) # (|V|, 64)
    gf2 = f if isinstance(net_city, nn.Identity) else net_city(f, H_city) # (|V|, 64)
    f = combiner(f, gf1, gf2) # (|V|, 64)

    pred = mlp(f).squeeze(-1) # (|V|)
    
    pred = pred[val_mask]
    ground_truth = Y[val_mask]

    mae = F.l1_loss(pred, ground_truth)
    mse = F.mse_loss(pred, ground_truth)

    return mae, mse, pred


if __name__ == '__main__':
    train(
        ablation=config.ablation,
        tcn=tcn,
        net_ind=net_ind,
        net_city=net_city,
        combiner=combiner,
        mlp=mlp,
        num_epochs=epochs,
        print_every=config.logging.print_every,
        gen_chart=config.chart.gen_chart
    )