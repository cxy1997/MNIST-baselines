class trainer(object):
    def __init__(self, data, model, config):
        super(trainer, self).__init__()
        self.data = data
        self.model = model
        self.config = config

    def train(self):
        print('train')
        