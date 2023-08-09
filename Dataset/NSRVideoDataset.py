from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch
from tqdm import tqdm
from enum import Enum
import numpy as np
import glob
import cv2
import math
import random
import os
import shutil

class DatasetType(Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"

class NSRVideoDataset(Dataset):
    def __init__(self, dataset_path : str, split : tuple, dataset_type : str, feature : str, use_frame_df : bool, is_transform : bool,
                    is_voting : bool = False, is_preprocess : bool = False) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        self.is_transform = is_transform
        self.feature = feature
        self.use_frame_df = use_frame_df
        self.is_voting = is_voting

        self.label_paths = glob.glob(dataset_path + "\*")
        agree_paths = glob.glob(self.label_paths[0] + "\*.mp4")
        agree_paths = [(agree_path, [1]) for agree_path in agree_paths]
        
        non_agree_paths = glob.glob(self.label_paths[1] + "\*.mp4")
        non_agree_paths = [(non_agree_path, [0]) for non_agree_path in non_agree_paths]

        train_split_index_agree, train_split_index_non_agree = int(len(agree_paths)*split[0]), int(len(non_agree_paths)*split[0])
        validation_split_index_agree, validation_split_index_non_agree = train_split_index_agree + int(len(agree_paths)*split[1]), train_split_index_non_agree + int(len(non_agree_paths)*split[1])

        random.shuffle(agree_paths)
        random.shuffle(non_agree_paths)

        if dataset_type == DatasetType.TRAIN.value:
            self.data_paths = agree_paths[:train_split_index_agree] + non_agree_paths[:train_split_index_non_agree]
        elif dataset_type == DatasetType.VALIDATION.value:
            self.data_paths = agree_paths[train_split_index_agree:validation_split_index_agree] + non_agree_paths[train_split_index_non_agree:validation_split_index_non_agree]
        elif dataset_type == DatasetType.TEST.value:
            if is_voting: # 전체 테스트셋 세그멘테이션을 위한 데이터셋 세팅
                # self.save_dir = r'D:\Video-Dataset\2022-NSR-voting-diag-thres' # 적절한 thres_value를 찾기 위한 dataset
                self.save_dir = r'D:\Video-Dataset\2023-NSR-voting-diag-test' # 확장 실험을 위한 dataset

                if is_preprocess:
                    print('Preprocessing of voting_test dataset, this will take long, but it will be done only once.')
                    print(f'{self.save_dir}: default save path\n{dataset_path} : default dataset path')
                    self.preprocess(dataset_path)
                
                print('load of voting_test dataset)')

                self.label_paths = glob.glob(self.save_dir + "\*")
                agree_paths = glob.glob(self.label_paths[0] + "\*\*")
                agree_paths = [(img_path, [1]) for img_path in agree_paths]

                non_agree_paths = glob.glob(self.label_paths[1] + "\*\*")
                non_agree_paths = [(img_path, [0]) for img_path in non_agree_paths]

                self.data_paths = agree_paths + non_agree_paths
                print(f'data_paths: {len(agree_paths)} + {len(non_agree_paths)} = {len(self.data_paths)}')

            else: # 전체 테스트셋 세그멘테이션이 아니면 일반 테스트셋 로드와 같음
                self.data_paths = agree_paths[validation_split_index_agree:] + non_agree_paths[validation_split_index_non_agree:]
                
        random.shuffle(self.data_paths)
            
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        video_diagonal = self.extract_feature(self.data_paths[index][0])
        label = torch.FloatTensor(self.data_paths[index][1])

        if self.is_transform:
            if self.feature == "hist":
                video_diagonal = self.transform_features(video_diagonal)
            elif self.feature == "diag":
                video_diagonal_tr = self.transform_features(video_diagonal.copy())
                mean, std = video_diagonal_tr.mean([1,2]), video_diagonal_tr.std([1,2])
                video_diagonal = self.transform_features(video_diagonal, mean, std, is_normalize=True)

        return video_diagonal, label
        

    def extract_feature(self, data_path : str):
        videocap = cv2.VideoCapture(data_path)
        video_total_frames_num = videocap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_frame_per_s = int(videocap.get(cv2.CAP_PROP_FPS))

        # Use frame difference
        if self.use_frame_df:
            section_split = 257
        # Use each frame 
        else:
            section_split = 256
        
        sections, retstep = np.linspace(1, video_total_frames_num, section_split, retstep=True)
        sections = list(map(math.floor, sections))
        frame_diagonals = []
        video_name = data_path.split("\\")[-1]
        
        while(videocap.isOpened()):
            ret, frame = videocap.read()
            
            if not ret:
                break
            
            if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) in sections and self.feature == "diag":
                frame = cv2.resize(frame, (256, 256))

                if self.use_frame_df:
                    if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
                        pre_frame = frame
                        continue
                
                    frame_df = cv2.absdiff(pre_frame, frame)
                    frame_r, frame_g, frame_b = frame_df[:,:,0], frame_df[:,:,1], frame_df[:,:,2]
                    pre_frame = frame
                
                else:
                    frame_r, frame_g, frame_b = frame[:,:,0], frame[:,:,1], frame[:,:,2]

                frame_r, frame_g, frame_b = np.diag(frame_r), np.diag(frame_g), np.diag(frame_b)
                frame_diagonal = np.stack([frame_r, frame_g, frame_b], -1)
                frame_diagonal = np.expand_dims(frame_diagonal, 1)
                frame_diagonals.append(frame_diagonal)
            
            elif int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) in sections and self.feature == "hist":
                frame_r, frame_g, frame_b = frame[:,:,0], frame[:,:,1], frame[:,:,2]

                frame_r_ht = cv2.calcHist(frame_r, [0], None, [256], [0, 256])
                frame_g_ht = cv2.calcHist(frame_g, [0], None, [256], [0, 256])
                frame_b_ht = cv2.calcHist(frame_b, [0], None, [256], [0, 256])

                if self.use_frame_df:
                    if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
                        pre_frame_r_ht = frame_r_ht
                        pre_frame_g_ht = frame_g_ht
                        pre_frame_b_ht = frame_b_ht
                        continue

                    frame_r_ht_df = abs(pre_frame_r_ht - frame_r_ht)
                    frame_g_ht_df = abs(pre_frame_g_ht - frame_g_ht)
                    frame_b_ht_df = abs(pre_frame_b_ht - frame_b_ht)

                    frame_diagonal = np.stack([frame_r_ht_df, frame_g_ht_df, frame_b_ht_df], -1)
                    pre_frame_r_ht = frame_r_ht
                    pre_frame_g_ht = frame_g_ht
                    pre_frame_b_ht = frame_b_ht

                else:
                    frame_diagonal = np.stack([frame_r_ht, frame_g_ht, frame_b_ht], -1)
                
                frame_diagonals.append(frame_diagonal)

        videocap.release()

        # use histogram feature with commercial log
        if self.feature == "hist":
            video_diagonal = np.concatenate(frame_diagonals, axis=1)
            video_diagonal = np.log(video_diagonal + 1)
            # video_diagonal = video_diagonal.astype(np.uint8)
            # video_diagonal = Image.fromarray(video_diagonal)
        elif self.feature == "diag":
            video_diagonal = np.concatenate(frame_diagonals, axis=1)

        return video_diagonal

    def preprocess(self, root_dir : str):
        error_list = open('runs-voting/error_list/list.txt', 'w')

        error_video = []
        
        for class_folder in os.listdir(root_dir):
            file_list = os.path.join(root_dir, class_folder)
            print(f'{class_folder}-videos are preprocessing...')

            for file in tqdm(os.listdir(file_list)):
                file_path = os.path.join(root_dir, class_folder, file)

                img_save_path = os.path.join(self.save_dir, class_folder, file[:-4])
                
                if not os.path.exists(img_save_path):
                    os.mkdir(img_save_path)
                
                videocap = cv2.VideoCapture(file_path)
                video_total_frames_num = videocap.get(cv2.CAP_PROP_FRAME_COUNT)
                video_frame_per_s = int(videocap.get(cv2.CAP_PROP_FPS))

                # Use frame difference
                if self.use_frame_df:
                    section_split = 257
                # Use each frame 
                else:
                    section_split = 256
        
                total_count = int(video_total_frames_num // (video_frame_per_s*180))
                start_frame = 1

                videocap.release()

                for idx in range(1, (total_count+1)):
                    frame_diagonals = []

                    sections = np.linspace(start_frame, video_frame_per_s*180*idx , section_split)
                    sections = list(map(math.floor, sections))

                    videocap = cv2.VideoCapture(file_path)
                    videocap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)

                    check_idx = 0
                    while(videocap.isOpened()):
                       
                        ret, frame = videocap.read()
                     
                        if not ret or (int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) == video_frame_per_s*180*idx+1) or check_idx == 256:
                            break
                        if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) in sections and self.feature == "diag":
                            frame = cv2.resize(frame, (256, 256))

                            if self.use_frame_df:
                                if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
                                    pre_frame = frame
                                    continue
                            
                                frame_df = cv2.absdiff(pre_frame, frame)
                                frame_r, frame_g, frame_b = frame_df[:,:,0], frame_df[:,:,1], frame_df[:,:,2]
                                pre_frame = frame
                            
                            else:
                                frame_r, frame_g, frame_b = frame[:,:,0], frame[:,:,1], frame[:,:,2]

                            frame_r, frame_g, frame_b = np.diag(frame_r), np.diag(frame_g), np.diag(frame_b)
                            frame_diagonal = np.stack([frame_r, frame_g, frame_b], -1)
                            frame_diagonal = np.expand_dims(frame_diagonal, 1)
                            frame_diagonals.append(frame_diagonal)
                            check_idx+=1
                        
                        elif int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) in sections and self.feature == "frame_diff_hist":
                            if int(videocap.get(cv2.CAP_PROP_POS_FRAMES)) == 1:
                                pre_frame = frame
                                continue

                            frame_df = cv2.absdiff(pre_frame, frame)
                            frame_r, frame_g, frame_b = frame_df[:,:,0], frame_df[:,:,1], frame_df[:,:,2]
                            pre_frame = frame

                            frame_r_ht = cv2.calcHist(frame_r, [0], None, [256], [0, 256])
                            frame_g_ht = cv2.calcHist(frame_g, [0], None, [256], [0, 256])
                            frame_b_ht = cv2.calcHist(frame_b, [0], None, [256], [0, 256])

                            frame_diagonal = np.stack([frame_r_ht, frame_g_ht, frame_b_ht], -1)
                            frame_diagonals.append(frame_diagonal)

                            check_idx += 1

                    start_frame = video_frame_per_s * 180 * idx + 1
                    
                    if self.feature == "diag":
                        try:
                            video_diagonal = np.concatenate(frame_diagonals, axis=1)

                            cv2.imwrite(f'./runs-voting/{idx}.jpg', video_diagonal)
                            shutil.move(f'./runs-voting/{idx}.jpg', img_save_path)
                        except:
                            error_video.append(file_path)

                    elif self.feature == "frame_diff_hist":
                        video_diagonal = np.concatenate(frame_diagonals, axis=1)
                        video_diagonal = np.log(video_diagonal + 1)
                        # video_diagonal = cv2.normalize(video_diagonal, None, 0, 255, cv2.NORM_MINMAX)
                        # video_diagonal = video_diagonal.astype(int)
                        try:
                            cv2.imwrite(f'./runs-voting/{idx}.jpg', video_diagonal)
                            shutil.move(f'./runs-voting/{idx}.jpg', img_save_path)
                        except:
                            error_video.append(file_path)

                for file_path in set(error_video):
                    error_list.write(file_path)

                videocap.release()
            
        print(f'{class_folder}-error_video : ', set(error_video))
        error_list.close()


    def imread(self, data_path : str):
        img_array = np.fromfile(data_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        return img
  
  
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