import pickle
import yaml
import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from dotmap import DotMap

from module.HGNNP import HGNNP
from module.combiner import WeightedFeatureCombiner, MLP
from module.TCN import TemporalConvNet
from module.generator import Generator3

with open("./configs/counterfactual.yaml", 'r') as f:
    config = DotMap(yaml.safe_load(f))

state_dict = torch.load(config.model_state_dict_path)

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

assert config.hypergraph == 'G_s' or config.hypergraph == 'G_i'
H = H_city if config.hypergraph == 'G_s' else H_industry

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

# Generator
generator = Generator3(H).to(device)

# Load state_dict
tcn.load_state_dict(state_dict['tcn_state_dict'])
net_ind.load_state_dict(state_dict['net_ind_state_dict'])
net_city.load_state_dict(state_dict['net_city_state_dict'])
combiner.load_state_dict(state_dict['combiner_state_dict'])
mlp.load_state_dict(state_dict['mlp_state_dict'])

tcn.eval()
net_ind.eval()
net_city.eval()
combiner.eval()
mlp.eval()
generator.train()

# Vertex feature of pretrained model
pretrained_feature = tcn(X.unsqueeze(1)).detach()

# Optimizer & lr-scheduler
epochs = config.epochs
params = list(generator.parameters())
optimizer = AdamW(
    params, 
    lr=config.lr,
    weight_decay=config.weight_decay,
)
lr_scheduler = CosineAnnealingLR(
    optimizer=optimizer, 
    T_max=epochs
)

pred = mlp(combiner(pretrained_feature, net_ind(pretrained_feature, H_industry), net_city(pretrained_feature, H_city))).squeeze(-1)
original_val_loss = F.l1_loss(pred[val_mask], Y[val_mask]).item()
original_train_loss = F.l1_loss(pred[train_mask], Y[train_mask]).item()
print(f"Original validation loss: {original_val_loss:.4f}")
print(f"Original training loss: {original_train_loss:.4f}")
B = original_train_loss

average_mae = 0
average_F = 0
average_loss = 0
lam = config.lam
print_every = config.logging.print_every

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    H_a = generator(H)                

    gf1 = net_ind(pretrained_feature, H_industry)         
    gf2 = net_city(pretrained_feature, H_a)        
    pred = combiner(pretrained_feature, gf1, gf2)
    pred = mlp(pred).squeeze(-1)       
        
    pred = pred[val_mask]
    ground_truth = Y[val_mask]

    mae_loss = -1 * (F.l1_loss(pred, ground_truth) - B)
    F_loss = torch.norm((H - H_a), p='fro')

    loss = mae_loss * lam + F_loss

    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    average_mae += mae_loss.item()
    average_F += F_loss.item()
    average_loss += loss.item()

    if epoch % print_every == 0:
        print(f"[epoch {epoch:03d}] mae_loss: {(average_mae / print_every):.4f}, F_loss: {(average_F / print_every):.4f}, total_loss:{(average_loss / print_every):.4f}")
        average_mae = 0
        average_F = 0
        average_loss = 0
        
        if epoch == epochs:
            print(f"[Last Epoch] mae_loss: {mae_loss.item():.4f}, F_loss: {F_loss.item():.4f}, total_loss:{loss.item():.4f}")
            os.makedirs(config.save_dir, exist_ok=True)
            torch.save(H_a, os.path.join(config.save_dir, f"H_a_{config.hypergraph}.pt"))
