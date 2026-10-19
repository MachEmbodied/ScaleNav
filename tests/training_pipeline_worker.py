"""CPU integration worker: real trainer with a small backbone and fixed text embeddings."""
import os
from pathlib import Path
import sys

import torch

from awm.training import recipe, runner


if __name__ == '__main__':
    torch.set_num_threads(1)
    recipe.CONFIG = Path(os.environ['AWM_TEST_RECIPE'])
    runner.require_training_device = lambda accelerator: None
    runner.load_text_encoder = lambda *args: (None, None)
    runner.encode_text = lambda text, *args: [torch.ones(3, 16) for _ in text]
    if os.environ.get('AWM_TEST_FAIL_STAGE'):
        save = runner.save_checkpoint

        def fail_save(accelerator, objective, output, phase, step, contract):
            if phase['stage'] == os.environ['AWM_TEST_FAIL_STAGE']:
                raise RuntimeError('Injected checkpoint failure')
            return save(accelerator, objective, output, phase, step, contract)

        runner.save_checkpoint = fail_save
    sys.exit(runner.main())
