import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, inputsize, outputsize) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputsize, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outputsize)
        )

    def forward(self, x):
        return self.net(x)


class DNN2(nn.Module):
    def __init__(self, inputsize, outputsize) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputsize, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outputsize)
        )

    def forward(self, x):
        return self.net(x)


class DNN3(nn.Module):
    def __init__(self, inputsize, outputsize) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputsize, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, outputsize)
        )

    def forward(self, x):
        return self.net(x)


class DNN4(nn.Module):
    def __init__(self, inputsize, outputsize) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputsize, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, outputsize)
        )

    def forward(self, x):
        return self.net(x)


class DNN5(nn.Module):
    def __init__(self, inputsize, outputsize) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputsize, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, outputsize)
        )

    def forward(self, x):
        return self.net(x)


class DNN_utilization(nn.Module):
    def __init__(self, inputsize, outputsize) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputsize, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, outputsize)
        )

    def forward(self, x):
        return self.net(x)
