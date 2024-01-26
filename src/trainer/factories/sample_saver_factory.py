from src.trainer.trainer_modules.binclass_task.jigsaw_sample_saver import JigsawSampleSaver
from src.trainer.trainer_modules.binclass_task.plain_sample_saver import PlainSampleSaver
from src.trainer.trainer_modules.patch_adg_task.patch_sample_saver import PatchSampleSaver


def get_sample_saver(params: dict):
    if params['name'] == 'jigsaw':
        return JigsawSampleSaver(params)

    elif params['name'] == 'patch_adjacence':
        return PatchSampleSaver(params)

    elif params['name'] == 'plain':
        return PlainSampleSaver(params)

    else:
        raise NotImplementedError(f'No savers found for task: {params["name"]}')
