import os
import tqdm
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils import data

np.random.seed(1234)
torch.manual_seed(1234)

TRAIN_SIZE = 0.8
NUM_PTS = 971
CROP_SIZE = 128
SUBMISSION_HEADER = "file_name,Point_M0_X,Point_M0_Y,Point_M1_X,Point_M1_Y,Point_M2_X,Point_M2_Y,Point_M3_X,Point_M3_Y,Point_M4_X,Point_M4_Y,Point_M5_X,Point_M5_Y,Point_M6_X,Point_M6_Y,Point_M7_X,Point_M7_Y,Point_M8_X,Point_M8_Y,Point_M9_X,Point_M9_Y,Point_M10_X,Point_M10_Y,Point_M11_X,Point_M11_Y,Point_M12_X,Point_M12_Y,Point_M13_X,Point_M13_Y,Point_M14_X,Point_M14_Y,Point_M15_X,Point_M15_Y,Point_M16_X,Point_M16_Y,Point_M17_X,Point_M17_Y,Point_M18_X,Point_M18_Y,Point_M19_X,Point_M19_Y,Point_M20_X,Point_M20_Y,Point_M21_X,Point_M21_Y,Point_M22_X,Point_M22_Y,Point_M23_X,Point_M23_Y,Point_M24_X,Point_M24_Y,Point_M25_X,Point_M25_Y,Point_M26_X,Point_M26_Y,Point_M27_X,Point_M27_Y,Point_M28_X,Point_M28_Y,Point_M29_X,Point_M29_Y\n"


class HorizontalFlip(object):
    def __init__(self, p=0.5, elem_name='image'):
        self.elem_name = elem_name
        self.p = p

    def __call__(self, sample):
        if torch.rand(1).item() < self.p and 'landmarks' in sample:
            sample[self.elem_name] = sample[self.elem_name][:, ::-1]
            # switching keypoint order
            sample['landmarks'][0:128] = sample['landmarks'][np.r_[64:128, 0:64]]
            sample['landmarks'][128:273] = sample['landmarks'][np.r_[272:127:-1]]
            sample['landmarks'][273:401] = sample['landmarks'][np.r_[337:401, 273:337]]
            sample['landmarks'][401:527] = sample['landmarks'][np.r_[464:527, 401:464]]
            sample['landmarks'][527:587] = sample['landmarks'][np.r_[586:526:-1]]
            sample['landmarks'][587:841] = sample['landmarks'][np.r_[714:841, 587:714]]
            sample['landmarks'][841:873] = sample['landmarks'][np.r_[872:840:-1]]
            sample['landmarks'][873:905] = sample['landmarks'][np.r_[904:872:-1]]
            sample['landmarks'][905:937] = sample['landmarks'][np.r_[936:904:-1]]
            sample['landmarks'][937:969] = sample['landmarks'][np.r_[968:936:-1]]
            sample['landmarks'][969:971] = sample['landmarks'][np.r_[970:968:-1]]
            # updating keypoints
            width = sample[self.elem_name].shape[1]
            sample['landmarks'][:, 0] = width - 1 - sample['landmarks'][:, 0]
        return sample


class ScaleMinSideToSize(object):
    def __init__(self, size=(CROP_SIZE, CROP_SIZE), elem_name='image'):
        self.size = torch.tensor(size, dtype=torch.float)
        self.elem_name = elem_name

    def __call__(self, sample):
        h, w, _ = sample[self.elem_name].shape
        if h > w:
            f = self.size[0] / w
        else:
            f = self.size[1] / h

        sample[self.elem_name] = cv2.resize(sample[self.elem_name], None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        sample["scale_coef"] = f

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2).float()
            landmarks = landmarks * f
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class CropCenter(object):
    def __init__(self, size=128, elem_name='image'):
        self.size = size
        self.elem_name = elem_name

    def __call__(self, sample):
        img = sample[self.elem_name]
        h, w, _ = img.shape
        margin_h = (h - self.size) // 2
        margin_w = (w - self.size) // 2
        sample[self.elem_name] = img[margin_h:margin_h + self.size, margin_w:margin_w + self.size]
        sample["crop_margin_x"] = margin_w
        sample["crop_margin_y"] = margin_h

        if 'landmarks' in sample:
            landmarks = sample['landmarks'].reshape(-1, 2)
            landmarks -= torch.tensor((margin_w, margin_h), dtype=landmarks.dtype)[None, :]
            sample['landmarks'] = landmarks.reshape(-1)

        return sample


class TransformByKeys(object):
    def __init__(self, transform, names):
        self.transform = transform
        self.names = set(names)

    def __call__(self, sample):
        for name in self.names:
            if name in sample:
                sample[name] = self.transform(sample[name])

        return sample


class FoldDatasetDataset(data.Dataset):
    def __init__(self, train_dataset, val_dataset, transforms, albu_transorms=None, split='train', fold=0, seed=42):
        super(FoldDatasetDataset, self).__init__()
        torch.manual_seed(seed)
        image_names = train_dataset.image_names + val_dataset.image_names
        image_names = np.array(image_names)
        landmarks = torch.cat((train_dataset.landmarks, val_dataset.landmarks))
        fold_indices = torch.randint(0, 5, (len(image_names),))
        if split == 'train':
            self.image_names = image_names[fold_indices != fold]
            self.landmarks = landmarks[fold_indices != fold]
        else:
            self.image_names = image_names[fold_indices == fold]
            self.landmarks = landmarks[fold_indices == fold]

        # the following data will be needed for albu augmentations
        self.min_xy = torch.min(torch.min(self.landmarks[:, :], dim=1).values, dim=1).values
        self.max_x = torch.max(self.landmarks[:, :, 0], dim=1).values
        self.max_y = torch.max(self.landmarks[:, :, 1], dim=1).values

        self.albu_transforms = albu_transorms
        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample['image'] = image

        if self.albu_transforms is not None:
            sample['keypoints'] = []
            try:
                sample = self.albu_transforms(image=image, keypoints=landmarks)
                # make sure no keypoints are removed by transforms
                assert len(sample['keypoints']) == NUM_PTS
                sample['landmarks'] = torch.tensor(np.stack(sample.pop('keypoints')))
            except:
                sample.pop('keypoints')
                sample['landmarks'] = landmarks
        else:
            sample['landmarks'] = landmarks

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


class ThousandLandmarksDataset(data.Dataset):
    def __init__(self, root, transforms, split="train"):
        super(ThousandLandmarksDataset, self).__init__()
        self.root = root
        landmark_file_name = os.path.join(root, 'landmarks.csv') if split is not "test" \
            else os.path.join(root, "test_points.csv")
        images_root = os.path.join(root, "images")

        self.image_names = []
        self.landmarks = []

        with open(landmark_file_name, "rt") as fp:
            num_lines = sum(1 for line in fp)
        num_lines -= 1  # header

        with open(landmark_file_name, "rt") as fp:
            for i, line in tqdm.tqdm(enumerate(fp)):
                if i == 0:
                    continue  # skip header
                if split == "train" and i == int(TRAIN_SIZE * num_lines):
                    break  # reached end of train part of data
                elif split == "val" and i < int(TRAIN_SIZE * num_lines):
                    continue  # has not reached start of val part of data
                elements = line.strip().split("\t")
                image_name = os.path.join(images_root, elements[0])
                self.image_names.append(image_name)

                if split in ("train", "val"):
                    landmarks = list(map(np.int16, elements[1:]))
                    landmarks = np.array(landmarks, dtype=np.int16).reshape((len(landmarks) // 2, 2))
                    self.landmarks.append(landmarks)

        if split in ("train", "val"):
            self.landmarks = torch.as_tensor(self.landmarks)
        else:
            self.landmarks = None

        self.transforms = transforms

    def __getitem__(self, idx):
        sample = {}
        if self.landmarks is not None:
            landmarks = self.landmarks[idx]
            sample["landmarks"] = landmarks

        image = cv2.imread(self.image_names[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        sample["image"] = image

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self):
        return len(self.image_names)


def restore_landmarks(landmarks, f, margins):
    dx, dy = margins
    landmarks[:, 0] += dx
    landmarks[:, 1] += dy
    landmarks /= f
    return landmarks


def restore_landmarks_batch(landmarks, fs, margins_x, margins_y):
    landmarks[:, :, 0] += margins_x[:, None]
    landmarks[:, :, 1] += margins_y[:, None]
    landmarks /= fs[:, None, None]
    return landmarks


def create_submission(path_to_data, test_predictions, path_to_submission_file):
    test_dir = os.path.join(path_to_data, "test")

    output_file = path_to_submission_file
    wf = open(output_file, 'w')
    wf.write(SUBMISSION_HEADER)

    mapping_path = os.path.join(test_dir, 'test_points.csv')
    mapping = pd.read_csv(mapping_path, delimiter='\t')

    for i, row in mapping.iterrows():
        file_name = row[0]
        point_index_list = np.array(eval(row[1]))
        points_for_image = test_predictions[i]
        needed_points = points_for_image[point_index_list].astype(np.int)
        wf.write(file_name + ',' + ','.join(map(str, needed_points.reshape(2 * len(point_index_list)))) + '\n')
