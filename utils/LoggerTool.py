from tensorboardX import SummaryWriter

class LoggerTool:
    def __init__(self, path):
        self.path = path
        self.writer = SummaryWriter(path)

    def write(self):
        pass