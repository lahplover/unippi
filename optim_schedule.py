'''A wrapper class for optimizer '''
import numpy as np


class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, n_warmup_steps, init_lr):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        # self.init_lr = np.power(d_model, -0.5)
        self.init_lr = init_lr
        self.steps_decay_scale = 10000
        # self.init_lr = 0.001

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        # return np.min([
        #     np.power(self.n_current_steps, -0.5),
        #     np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

        # return 0.9**(self.n_current_steps/self.steps_decay_scale)
        warmup_factor = np.min([1, self.n_current_steps / self.n_warmup_steps])
        return 0.95**(self.n_current_steps/self.steps_decay_scale) * warmup_factor**0.5
        # return 1.05**(self.n_current_steps/500)

        # return 1.0

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        # print('learning rate ', lr)

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
