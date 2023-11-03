import os
import wandb
import warnings
import torch


def download_model(name, run_id, entity, project, overwrite=False):
    if os.path.exists(f'./checkpoints/{name}.ckpt') and not overwrite:
        return f'./checkpoints/{name}.ckpt'
    api = wandb.Api()

    artifact = api.artifact(f'{entity}/{project}/model-{run_id}:v0')
    model_path = artifact.download('./checkpoints')
    # rename the model
    os.rename(f'{model_path}/model.ckpt', f'{model_path}/{name}.ckpt')
    return f'{model_path}/{name}.ckpt'


def model_from_file(model_path,
                    model_cls,
                    map_location_string: None | str = None):
    map_device = None
    if map_location_string is not None:
        map_device = torch.device(map_location_string)
    model = model_cls.load_from_checkpoint(model_path,
                                           strict=False,
                                           map_location=map_device)
    model.eval()
    return model


def filter_runs(runs, tags):
    for run in runs:
        if all(tag in run.tags for tag in tags):
            yield run


def get_models(entity,
               project,
               tags,
               model_cls,
               sort_by_name=True,
               map_location_string='cpu',
               overwrite=True):
    api = wandb.Api()
    runs = api.runs(path=f"{entity}/{project}")

    runs = list(filter_runs(runs, tags))
    if sort_by_name:
        runs = sorted(runs, key=lambda run: run.name)

    models = []
    for run in runs:
        artifacts = run.logged_artifacts()

        # find best artifact
        best_artifact = None
        for artifact in artifacts:
            if 'best_k' in artifact._aliases:
                best_artifact = artifact
                break

        if best_artifact is None:
            warnings.warn('No best artifact found for run: {}'.format(
                run.name))
            continue

        # Download the artifact
        if os.path.exists(f'./checkpoints/{run.name}.ckpt') and not overwrite:
            model_file = f'./checkpoints/{run.name}.ckpt'
        else:
            model_path = best_artifact.download('./checkpoints')
            model_file = f'{model_path}/{run.name}.ckpt'
            os.rename(f'{model_path}/model.ckpt', model_file)

        try:
            model = model_from_file(model_file,
                                    model_cls,
                                    map_location_string=map_location_string)
        except Exception as e:
            model = model_from_file(model_file,
                                    model_cls,
                                    map_location_string='cpu')

        models.append(model)

    return models
