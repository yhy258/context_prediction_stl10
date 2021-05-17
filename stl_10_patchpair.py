import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import numpy as np

class DatasetForPretext(Dataset):
    def __init__(self, ):
        self.dataset = datasets.STL10(root='./.data', split='unlabeled', download=True)  # 10000,96,96
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.gap = 96 // 2
        self.patch_size = 96
        self.jitter_size = 7

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x = self.dataset[idx][0]
        x = self.transform(x)  # b, 3, 448 448
        x = transforms.Resize([448, 448])(x)

        patches = {}

        channels_std = [0.22414584, 0.22148906, 0.22389975]

        remove_channel_idx = np.random.choice([0, 1, 2], 2, replace=False)  # 삭제할거 선택
        k = np.arange(1, 4)
        for i in remove_channel_idx:
            k = np.where(k == (i + 1), 0, k)
        remaining_channel = k[k != 0][0]  # 남은 채널 index로 사용하려면 -1 해야함.

        gaussian_std = channels_std[remaining_channel - 1] / 100
        for i in remove_channel_idx:
            x[i, :, :] = torch.zeros(1, 448, 448).data.normal_(0, gaussian_std)
        # 조건 : 가장 좌측의 경우 50 전까지에서 랜덤으로, 가장 높은 경우는 50 전까지 랜덤으로
        # 계산해보면 target 즉 가운데의 경우는 151~ 201 까지임 (왼상단 기준)
        target_coord_1 = np.random.randint(151, 202, size=2)
        # target_coord_2 = target_coord_1 + self.patch_size

        randi = np.random.randint(1, 9)  # label
        this_coord_1 = self.make_top_left_coord(randi, target_coord_1)
        # this_coord_2 = this_coord_1 + self.patch_size

        for i in range(2):
            jitter = np.random.randint(-self.jitter_size, self.jitter_size + 1)
            this_coord_1[i] += jitter

        center = x[:, int(target_coord_1[0]):int(target_coord_1[0]) + self.patch_size,
                 int(target_coord_1[1]):int(target_coord_1[1]) + self.patch_size]

        other = x[:, int(this_coord_1[0]):int(this_coord_1[0]) + self.patch_size,
                int(this_coord_1[1]):int(this_coord_1[1]) + self.patch_size]

        label = randi - 1
        return center, other, label  # 'center' : image, 'other' : image

    def make_top_left_coord(self, label, target_coord_1):
        this_coord = copy.deepcopy(target_coord_1)
        if label in [1, 4, 6]:
            this_coord[1] = target_coord_1[1] - self.patch_size
            this_coord[1] -= self.gap

        if label in [1, 2, 3]:
            this_coord[0] = target_coord_1[0] - self.patch_size
            this_coord[0] -= self.gap

        if label in [3, 5, 8]:
            this_coord[1] = target_coord_1[1] + self.patch_size
            this_coord[1] += self.gap

        if label in [6, 7, 8]:
            this_coord[0] = target_coord_1[0] + self.patch_size
            this_coord[0] += self.gap

        return this_coord