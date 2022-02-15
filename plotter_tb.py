from tensorboardX import SummaryWriter

class TensorboardLossPlotter(object):
    def __init__(self, env_name='main'):
        self.writer = SummaryWriter(log_dir=env_name)
        self.env = env_name
        self.plots = {}