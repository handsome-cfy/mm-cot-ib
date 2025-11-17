from transformers import TrainerCallback
import swanlab as wandb  # 确保使用的是正确的 wandb 库

class WandbCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        wandb.log({"train/begin": 1})

    def on_train_end(self, args, state, control, **kwargs):
        wandb.log({"train/end": 1})

    def on_epoch_begin(self, args, state, control, epoch=None, **kwargs):
        if epoch is not None:
            wandb.log({"train/epoch/begin": epoch})

    def on_epoch_end(self, args, state, control, epoch=None, **kwargs):
        if epoch is not None:
            wandb.log({"train/epoch/end": epoch})

    def on_step_begin(self, args, state, control, step=None, **kwargs):
        if step is not None:
            wandb.log({"train/step/begin": step})

    def on_step_end(self, args, state, control, step=None, **kwargs):
        if step is not None:
            wandb.log({"train/step/end": step})

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs)