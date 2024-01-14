from src.trainer.trainer_modules.jigsaw_sample_saver import JigsawSampleSaver
from src.trainer.trainer_modules.patch_sample_saver import PatchSampleSaver


def get_sample_saver(params: dict):
    if params['name'] == 'jigsaw':
        return JigsawSampleSaver(params)

    elif params['name'] == 'patch_adjacence':
        return PatchSampleSaver(params)
