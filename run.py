import os

os.environ["OMP_NUM_THREADS"] = "1"

from utils.commons.hparams import hparams, set_hparams
import importlib

def run_task():
    assert hparams['task_cls'] != ''
    pkg = ".".join(hparams["task_cls"].split(".")[:-1])     # pkg: tasks.tts.fs2_orig
    cls_name = hparams["task_cls"].split(".")[-1]    # cls_name: FastSpeech2OrigTask
    task_cls = getattr(importlib.import_module(pkg), cls_name)
    task_cls.start()    # tasks.tts.fs2_orig.FastSpeech2OrigTask 실행

if __name__ == '__main__':
    set_hparams()
    run_task()
