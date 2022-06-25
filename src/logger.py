import sys


class Logger:
    def __init__(self, file=None):
        self.file = file

    def logging(self, *logs):
        if self.file:
            with open(self.file, 'a+') as f:
                for log in logs:
                    f.write(log)
                    f.write('\n')
        else:
            for log in logs:
                sys.stdout.write(log)
                sys.stdout.write('\n')