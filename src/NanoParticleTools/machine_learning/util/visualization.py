from matplotlib import pyplot as plt
import torch
from torch.nn.functional import mse_loss


def get_predictions(model, dataloader, log=False, log_constant=10):
    uv_pred = []
    uv_true = []
    for data in dataloader:
        out = model.predict_step(data)
        if log:
            uv_pred.append(out.detach())
            uv_true.append(data.log_y)
        else:
            uv_pred.append(torch.pow(10, out.detach()) - log_constant)
            uv_true.append(data.y)
    uv_pred = torch.cat(uv_pred)
    uv_true = torch.cat(uv_true)
    return uv_pred, uv_true


def get_parity_plot(model,
                    data_module,
                    ax=None,
                    log=False,
                    log_constant=10,
                    fig_kwargs={}):
    if ax is None:
        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)

    metrics = {}
    if data_module.train_dataset is not None:
        uv_pred, uv_true = get_predictions(model,
                                           data_module.train_dataloader(), log,
                                           log_constant)
        ax.plot(uv_true.flatten(),
                uv_pred.flatten(),
                'o',
                alpha=0.2,
                label='Train Data',
                color='tab:blue')
        metrics['train_mse'] = mse_loss(uv_pred, uv_true)

    if data_module.val_dataset is not None:
        uv_pred, uv_true = get_predictions(model, data_module.val_dataloader(),
                                           log, log_constant)
        ax.plot(uv_true.flatten(),
                uv_pred.flatten(),
                'X',
                alpha=0.2,
                label='Val Data',
                color='tab:orange')
        metrics['val_mse'] = mse_loss(uv_pred, uv_true)

    if data_module.iid_test_dataset is not None:
        uv_pred, uv_true = get_predictions(model,
                                           data_module.iid_test_dataloader(),
                                           log, log_constant)
        ax.plot(uv_true.flatten(),
                uv_pred.flatten(),
                'D',
                alpha=0.2,
                label='ID Test Data',
                color='tab:green')
        metrics['id_test_mse'] = mse_loss(uv_pred, uv_true)

    if data_module.test_dataset is not None:
        uv_pred, uv_true = get_predictions(model,
                                           data_module.test_dataloader(), log,
                                           log_constant)
        ax.plot(uv_true.flatten(),
                uv_pred.flatten(),
                'D',
                alpha=0.2,
                label='OOD Test Data',
                color='tab:red')
        metrics['ood_test_mse'] = mse_loss(uv_pred, uv_true)

    ax.plot([0, max(max(ax.get_xlim()), max(ax.get_ylim()))],
            [0, max(max(ax.get_xlim()), max(ax.get_ylim()))], 'k--')

    ax.set_xlabel('NPMC UV Intensity', fontsize=18)
    ax.set_ylabel('ML Predicted UV Intensity', fontsize=18)
    ax.legend(fontsize=11)
    return metrics
