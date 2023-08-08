import os
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from PIL import Image
import time
from torch.utils.data import Dataset
from Models.Densenet201 import Densenet201

# 파일 및 폴더 경로
image_folder = r'C:\Users\VIP444\Pictures\2023-hist-test'
result_file = f"./runs-seg-test/{time.strftime('%Y-%m-%d-%H%M%S')}.txt"
model_folder = r'C:\Users\VIP444\Documents\hist\VideoClassification\runs\2023-07-17-090248-Densenet201-diagnal_log\models'
model_filename = 'checkpoint.pth.tar'


# 모델 불러오기
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Densenet201().to(device, non_blocking=True)
criterion = torch.nn.BCEWithLogitsLoss().to(device, non_blocking=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

load_path = os.path.join(model_folder, model_filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

load_checkpoint(torch.load(load_path), model, optimizer)
model.eval()
model.to('cpu')


# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path)

        image_tr = self.transform_features(image.copy())
        mean, std = image_tr.mean([1,2]), image_tr.std([1,2])
        image = self.transform_features(image, mean, std, is_normalize=True)

        return image
    
    def transform_features(self, image, mean = 0, std = 0, is_normalize = False):
        if is_normalize:
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)
            ])

            return transform(image)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(), 
            ])

            return transform(image)


# 커스텀 데이터셋 인스턴스 생성
test_dataset = CustomDataset(image_folder)

# 데이터로더 설정
batch_size = 64
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 결과 처리 및 파일 저장
with open(result_file, 'w', encoding='utf-8') as result_file:
    result_file.write("original;predicted;name\n")
    with torch.no_grad():
        for images in test_dataloader:
            images = images.to('cpu')
            outputs = model(images)
            predicted_classes = (outputs > 0.5).int().tolist()

            for i, result in enumerate(predicted_classes):
                image_name = test_dataset.image_files[i]
                org_val = res_val = 'agr'
                if result == [0]:
                    res_val = 'non'

                print(f"{image_name[:3]};{res_val};{image_name[4:]}")
                result_file.write(f"{image_name[:3]};{res_val};{image_name[4:]}\n")
