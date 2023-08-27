import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
import random
import matplotlib.pyplot as plt
import pyautogui as pag
import PIL
import torchvision.transforms as transforms
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import time

pag.FAILSAFE = False

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
writer = SummaryWriter()
train_time = 0


# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, num_actions):
        print(f"0 / {time.time()}")
        super(DQN, self).__init__()
        print(f"1 / {time.time()}")
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True))
        print(f"2 / {time.time()}")
        self.conv2 = nn.Sequential(nn.Conv2d(16, 64, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        print(f"3 / {time.time()}")
        self.conv3 = nn.Sequential(nn.Conv2d(64, 256, 3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(inplace=True))
        print(f"4 / {time.time()}")
        self.actor_linear = nn.Sequential(nn.Linear(1851392, 256),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(256, num_actions))
        print(f"5 / {time.time()}")

    def forward(self, x):  # 각 action에 대한 가치를 softmax를 거쳐서 확률로 출력
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.actor_linear(x.reshape(x.size(0), -1))
        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 10)
        else:
            return out.argmax().item()

action_list = [[2720, 550], [2860, 550], [3000, 550], [2720 ,660], [2860, 660], [3000, 660], [2765, 900], [2855, 900], [2950, 900], [3055, 900], [-1, -1]]
#action list 정의. action list는 6개의 필드 위치와 4개의 카드 위치 그리고 한개의 아무것도 안하는 리워드를 주었다.
action_list_name = {0:'left top', 1:'center top', 2:'right top', 3:'right bottom', 4:'center bottom', 5:'center right', 6:'card 1', 7:'card 2', 8:'card 3', 9:'card 4', 10:'rest action'}


class ENV():
    def __init__(self):
        # screenshot의 위치 지정, 클래스 생성할때 가져오기

        # winflg와 lose flag 존재해야 함. 0으로 하는 건 grayscale
        self.winFlag = cv2.imread('/home/jnu/Desktop/Reinforce/Royale/screenshot/cr_test/weWin.png', 0)
        self.loseFlag = cv2.imread('/home/jnu/Desktop/Reinforce/Royale/screenshot/cr_test/enemyWin.png', 0)
        # nocard flag
        self.nocardFlag = cv2.imread('/home/jnu/Desktop/Reinforce/Royale/screenshot/test1/nocard.png')
        # noElixir flag
        self.noelixirFlag = cv2.imread('/home/jnu/Desktop/Reinforce/Royale/screenshot/test1/noElixir.png')
        self.startGameFlag = cv2.imread('/home/jnu/Desktop/Reinforce/Royale/screenshot/cr_test/battleStart.png')

    def return_state(self, img):
        # 스크린 샷을 인자로 받아와서 모델에 넣을 수 있도록 tensor로 변환
        tf = transforms.ToTensor()
        img_t = tf(img)
        img_t = img_t.unsqueeze(0)
        # img_t = img_t.permute(1, 0, 2, 3)

        return img_t

    def check_win(self, img):
        # 게임이 이겼는지 확인, screenshot을 가져와서 우리가 원하는 크기로 잘라서 확인
        # img = np.array(img)
        checkFlag1 = np.array(img.crop((225, 335, 280, 365)))
        checkFlag1 = cv2.cvtColor(checkFlag1, cv2.COLOR_BGR2GRAY)
        win_check = cv2.matchTemplate(checkFlag1, self.winFlag, cv2.TM_CCOEFF_NORMED)
        if win_check > 0.8:
            return 1

        else:
            return 0

    def check_lose(self, img):
        # 게임이 졌는지 확인, screenshot을 가져와서 우리가 원하는 크기로 잘라서 확인
        # img = np.array(img)
        checkFlag2 = np.array(img.crop((225, 85, 280, 115)))
        checkFlag2 = cv2.cvtColor(checkFlag2, cv2.COLOR_BGR2GRAY)
        lose_check = cv2.matchTemplate(checkFlag2, self.loseFlag, cv2.TM_CCOEFF_NORMED)
        if lose_check > 0.8:
            return 1

        else:
            return 0

    def check_card(self, img):
        # 카드를 선택하지 않았는지 확인, screenshot을 가져와서 init에 지정된 nocard 이미지와 비교하여 reward 부여
        nocard = cv2.cvtColor(self.nocardFlag, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ratio = cv2.matchTemplate(nocard, img, cv2.TM_CCOEFF_NORMED)

        if (np.max(ratio) > 0.90):
            # print(np.max(ratio))
            return 1

        else:
            return 0

    def check_elixir(self, img):
        noElixir = cv2.cvtColor(self.noelixirFlag, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ratio = cv2.matchTemplate(noElixir, img, cv2.TM_CCOEFF_NORMED)

        if (np.max(ratio) > 0.90):
            # print(np.max(ratio))
            return 1

        else:
            return 0

    def checkET1(self, img):
        score1 = 0
        checkFlag1 = np.array(img.crop((99, 135, 149, 136)))

        for i in range(50):
            if (checkFlag1[0][i][0] <= 96):
                score1 += 1

        score1 = score1 * 2
        return score1

    def checkET2(self, img):
        score2 = 0
        checkFlag2 = np.array(img.crop((356, 135, 406, 136)))

        for i in range(50):
            if (checkFlag2[0][i][0] <= 96):
                score2 += 1

        score2 = score2 * 2
        return score2

    def checkGameStart(self, img):
        startMessage = cv2.cvtColor(self.startGameFlag, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ratio = cv2.matchTemplate(startMessage, img, cv2.TM_CCOEFF_NORMED)
        if (np.max(ratio) > 0.90):
            # print(np.max(ratio))
            return 1

        else:
            return 0

    # def checkET3

    # def checkOT1
    # def checkOT1
    # def checkOT1

    # 우리 타워와 상대 타워의 hp를 확인하여 reward 부여

    def retryGame(self):
        pag.click((2860, 875))
        time.sleep(2)
        pag.click((3070, 185))
        time.sleep(1)
        pag.click((2920, 385))
        time.sleep(0.5)
        pag.click((2950, 615))


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque()
        self.batch_size = 5
        self.size_limit = 50000

    def put(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()

    def sample(self, n):
        print(f"self.buffer = {len(self.buffer)}")
        return random.sample(self.buffer, n)

    def size(self):
        return len(self.buffer)


def train(q, q_target, memory, gamma, optimizer, batch_size):
    global train_time
    for i in range(3):
        # print(f"batch_size = {batch_size}, memory.size = {memory.size()}")
        batch = memory.sample(batch_size)
        # print(f"size = {len(batch[0])}")
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_lst = torch.stack(s_lst)
        s_prime_lst = torch.stack(s_prime_lst)
        # print(f"about s = {type(s_lst)}")
        s, a, r, s_prime, done_mask = s_lst, torch.tensor(a_lst), \
            torch.tensor(r_lst), s_prime_lst, \
            torch.tensor(done_mask_lst)

        # print(f"about S = {type(s)}")
        s = s.squeeze()
        s_prime = s_prime.squeeze()
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(target, q_a)
        print(f"loss = {loss}")

        train_time += 1
        writer.add_scalar("Loss/train", loss, train_time)
        writer.flush()

        optimizer.zero_grad()
        print("we got the end1")
        loss.backward()
        print("we got the end2")
        optimizer.step()
        print("we got the end3")


env = ENV()
q = DQN(len(action_list))
q_target = DQN(len(action_list))
q_target.load_state_dict(q.state_dict())
memory = ReplayBuffer()

avg_t = 0
gamma = 0.98
batch_size = 20
optimizer = optim.Adam(q.parameters(), lr = 0.0005)

middleBuffer = []

n_epi = 0
done = 0
import time

while (1):
    # s 변수 설정
    img = pag.screenshot(region=(2605, 100, 510, 900))
    img_t = env.return_state(img)
    s = img_t

    # reward 초기화
    reward = 0
    # enemy tower score / score1 = left, score2 = right
    score1 = 100
    score2 = 100
    t1 = time.time()
    start_flag = 0
    while (1):
        while (start_flag == 0):
            # 스크린샷 찍기
            img = pag.screenshot(region=(2605, 100, 510, 900))

            if (env.checkGameStart(img)):
                # print("game start")
                time.sleep(3)
                start_flag = 1
                break
        # epsilon 조절
        epsilon = max(0.1, 0.5 - 0.01 * (train_time / 200))
        n_epi += 1
        reward -= 1

        # 스크린샷 찍기
        img = pag.screenshot(region=(2605, 100, 510, 900))

        # 모델에 넣기 위해 tensor로 변환
        img_t = env.return_state(img)
        s_prime = img_t
        a = q.sample_action(img_t, epsilon)
        if(a!=10):
            # 화면 클릭
            pag.click(action_list[a][0], action_list[a][1])


        # 승리 확인
        if (env.check_win(img)):
            print("win")
            reward += 10000
            done = 1


        # 패배 확인
        elif (env.check_lose(img)):
            print("lose")
            reward -= 10000
            done = 1

        # no card확인
        if (env.check_card(img)):
            reward -= 100
            print("no card")

        # no elixir확인
        if (env.check_elixir(img)):
            reward -= 100
            print("no Elixir")

        # enemy tower reward calculate
        score1_now = env.checkET1(img)
        score2_now = env.checkET1(img)
        # print(f"score1 = {score1}, score1_now = {score1_now}")
        # print(f"score2 = {score2}, score2_now = {score2_now}")
        if (score1_now < score1):
            reward += 10 * (score1 - score1_now)
            score1 = score1_now

        if (score2_now < score2):
            reward += 10 * (score2 - score2_now)
            score2 = score2_now

        middleBuffer.append((s, a, reward, s_prime, done))
        print('action =', action_list_name[a], '    index = ', action_list[a][0], action_list[a][1], '    reward = ',
              reward)

        s = s_prime

        #속도조절
        time.sleep(0.3)
        
        writer.add_scalar("reward/epi", reward, n_epi)
        writer.flush()

        if done == 1:
            break

    # memory size 2000넘으면 학습 시작
    if len(middleBuffer) >= 20:
        print(time.time() - t1)
        for i in range(len(middleBuffer)):
            memory.put(middleBuffer[i])

        print(len(memory.buffer))
        print("========================")
        # print(middleBuffer.size())
        middleBuffer = []

        train(q, q_target, memory, gamma, optimizer, batch_size)

        print(f"train time = {train_time}")
        q_target.load_state_dict(q.state_dict())
        print(f"epi = {n_epi}, buffer size = {memory.size()}, epsilon = {epsilon}")

    # reak
    env.retryGame()
    time.sleep(5)

