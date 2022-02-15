from tensorboardX import SummaryWriter

class TensorboardPlotter(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir) # log가 저장될 경로

    def loss_plot(self, var_name, split_name, loss_name, x, y):
        self.writer.add_scalar(var_name+'/'+split_name,
                                {loss_name, x}, 
                                y)

    def img_plot(self, title, img, iter):
        self.writer.add_image(title.split[' '][0]+'/'+title,
                                img, 
                                iter)

    def close(self):
        self.writer.close()