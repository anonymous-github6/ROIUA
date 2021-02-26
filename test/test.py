import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset
from models.backbone import Backbone
from models.head import ROIUA


if __name__ == '__main__':

    DATA_ROOT = "./imgs"
    BACKBONE_RESUME_PATH = './checkPoints/backbone_ir50_ms1m_epoch120.pth'
    HEAD_RESUME_PATH = './checkPoints/ROIUA_epoch4.pth'
    score_file = open('./result/score.txt', 'w')
    INPUT_SIZE = [112, 112]
    BATCH_SIZE = 32

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize([112,112]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

	##########feat extract data fetch
    dataset= datasets.ImageFolder(DATA_ROOT, transform)
    print(dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle = False)

	#########backbone
    BACKBONE = Backbone([112, 112], 50, 'ir')
    if os.path.isfile(BACKBONE_RESUME_PATH):
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_PATH), strict=True)
    else:
        print("No BACKBONE Checkpoint Found")
    BACKBONE = BACKBONE.to(DEVICE)
    BACKBONE.eval()  # set to training mode

	########our method
    HEAD = ROIUA()
    if os.path.isfile(HEAD_RESUME_PATH):
        HEAD.load_state_dict(torch.load(HEAD_RESUME_PATH), strict=True)
    else:
        print("No HEAD Checkpoint Found")
    HEAD = HEAD.to(DEVICE)
    HEAD.eval()

    batch = 0
    for inputs,_ in loader:
        print('processing batch:{}'.format(batch))
        with torch.no_grad():
            inputs = inputs.to(DEVICE)
            feature = BACKBONE(inputs)
            score = HEAD(feature)
            if len(score.size()) > 1:
                score = score.squeeze()
            if torch.cuda.is_available():
                score = score.data.cpu().numpy()
            else:
                score = score.numpy()
            for idx in range(score.shape[0]):
                score_file.write(dataset.imgs[idx][0] + ' ' + str(score[idx]) + '\n')
    score_file.close()