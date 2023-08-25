import os
import wandb


def download_model(name, run_id, entity, project, overwrite=False):
    if os.path.exists(f'./checkpoints/{name}.ckpt') and not overwrite:
        return f'./checkpoints/{name}.ckpt'
    api = wandb.Api()

    artifact = api.artifact(f'{entity}/{project}/model-{run_id}:v0')
    model_path = artifact.download('./checkpoints')
    # rename the model
    os.rename(f'{model_path}/model.ckpt', f'{model_path}/{name}.ckpt')
    return f'{model_path}/{name}.ckpt'


def model_from_file(model_path, model_cls, map_location_string):
    model = model_cls.load_from_checkpoint(model_path, map_location=torch.device(map_location_string))
    model.eval()
    return model
