import torch
import torch.nn as nn
import torch.nn.functional as F


class PPO(nn.Module):
    def __init__(self, num_actions):
        super(PPO, self).__init__()

        self.data = []

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1),
                                   # nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.actor_linear = nn.Sequential(nn.Linear(256, 128),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(128, num_actions))
        self.critic_linear = nn.Sequential(nn.Linear(256, 128),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(128, 1))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim=0):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.actor_linear(x.reshape(x.size(0), -1))
        x = self.avgpool(x).reshape(x.size(0), -1)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x).reshape(x.size(0), -1)
        v = self.critic_linear(x.reshape(x.size(0), -1))
        return v


if __name__ == '__amin__':
    a = torch.rand(2, 3, 510, 900).cuda()
    model = PPO(num_actions=45).cuda()
    print(model.v(a).shape)