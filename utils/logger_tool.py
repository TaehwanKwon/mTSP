from tensorboardX import SummaryWriter

class LoggerTool:
    def __init__(self, path):
        self.path = path
        self.writer = SummaryWriter(path)

    def write(self, step, stats):
    	for key in stats:
    		self.writer.add_scalar(f"{key}", step, stats[key])

