import inspect
import os
import runpy

import pytorch_lightning as pl

orig_fit = pl.Trainer.fit


def fit_with_resume(self, *args, **kwargs):
    ckpt = os.environ.get('SC_RESUME_CKPT', '').strip()
    if ckpt:
        sig = inspect.signature(orig_fit)
        if 'ckpt_path' in sig.parameters:
            kwargs.setdefault('ckpt_path', ckpt)
            print(f"[resume-wrapper] injecting ckpt_path={ckpt}")
        else:
            # Backward compatibility for older PL versions.
            if not getattr(self, 'resume_from_checkpoint', None):
                self.resume_from_checkpoint = ckpt
                print(f"[resume-wrapper] setting resume_from_checkpoint={ckpt}")
    return orig_fit(self, *args, **kwargs)


pl.Trainer.fit = fit_with_resume
runpy.run_path('tutorial_train.py', run_name='__main__')
