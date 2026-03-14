import torch
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def save_best(epoch, tcn, net_ind, net_city, combiner, mlp, save_dir):
    torch.save({
        'epoch': epoch,
        'tcn_state_dict': tcn.state_dict(),
        'net_ind_state_dict': net_ind.state_dict(),
        'net_city_state_dict': net_city.state_dict(),
        'combiner_state_dict': combiner.state_dict(),
        'mlp_state_dict': mlp.state_dict(),
    }, f"{save_dir}/checkpoints/model_best.pt")


def save_epoch(epoch, tcn, net_ind, net_city, combiner, mlp, save_dir):
    torch.save({
        'epoch': epoch,
        'tcn_state_dict': tcn.state_dict(),
        'net_ind_state_dict': net_ind.state_dict(),
        'net_city_state_dict': net_city.state_dict(),
        'combiner_state_dict': combiner.state_dict(),
        'mlp_state_dict': mlp.state_dict(),
    }, f"{save_dir}/checkpoints/epoch{epoch}.pt")


def plot_loss(train_loss, val_loss, interval, save_path, set_max_val: int = None, set_min_val: int = None):
    train_loss = [x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in train_loss]
    val_loss = [x.detach().cpu().item() if torch.is_tensor(x) else float(x) for x in val_loss]
    cmb = train_loss + val_loss
    max_val = max(cmb) if set_max_val is None else set_max_val
    min_val = min(cmb) if set_min_val is None else set_min_val

    epochs_train = list(range(0, len(train_loss) * interval, interval))
    epochs_val = list(range(0, len(val_loss) * interval, interval))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    axes[0].plot(epochs_train, train_loss, label='Train Loss', color='blue')
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_ylim(bottom=min_val, top=max_val)
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs_val, val_loss, label='Validation Loss', color='orange')
    axes[1].set_title('Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_ylim(bottom=min_val, top=max_val)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()