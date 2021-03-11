import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF
import torchvision.transforms as transforms

landmarks = ["x_0", "x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7", "x_8", "x_9",
                            "x_10", "x_11", "x_12", "x_13", "x_14", "x_15", "x_16", "x_17", "x_18", "x_19", "x_20",
                            "x_21", "x_22", "x_23", "x_24", "x_25", "x_26", "x_27", "x_28", "x_29", "x_30", "x_31",
                            "x_32", "x_33", "x_34", "x_35", "x_36", "x_37", "x_38", "x_39", "x_40", "x_41", "x_42",
                            "x_43", "x_44", "x_45", "x_46", "x_47", "x_48", "x_49", "x_50", "x_51", "x_52", "x_53",
                            "x_54", "x_55", "x_56", "x_57", "x_58", "x_59", "x_60", "x_61", "x_62", "x_63", "x_64",
                            "x_65", "x_66", "x_67", "y_0", "y_1", "y_2", "y_3", "y_4", "y_5", "y_6", "y_7", "y_8",
                            "y_9", "y_10", "y_11", "y_12", "y_13", "y_14", "y_15", "y_16", "y_17", "y_18", "y_19",
                            "y_20", "y_21", "y_22", "y_23", "y_24", "y_25", "y_26", "y_27", "y_28", "y_29", "y_30",
                            "y_31", "y_32", "y_33", "y_34", "y_35", "y_36", "y_37", "y_38", "y_39", "y_40", "y_41",
                            "y_42", "y_43", "y_44", "y_45", "y_46", "y_47", "y_48", "y_49", "y_50", "y_51", "y_52",
                            "y_53", "y_54", "y_55", "y_56", "y_57", "y_58", "y_59", "y_60", "y_61", "y_62", "y_63",
                            "y_64", "y_65", "y_66", "y_67"]

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    #
    # @property
    # def num_frames(self):
    #     return int(self._data[3] - self._data[2])

    @property
    def min_frame(self):
        return int(self._data[2])

    @property
    def max_frame(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, mode,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        # self.root_path = root_path
        # self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self.db_path = "/gpu-data/filby/EmoReact_V_1.0/Data"

        self.categorical_emotions = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]

        self.continuous_emotions = ["Valence"]
        
        self.df = pd.read_csv(os.path.join("EmoReact/{}.csv".format(mode)))

        self.video_list = self.df["video"]
        self.mode = mode


    def get_bounding_box(self, image, keypoints, format="cv2"):

        keypoints = keypoints.values
        # display(keypoints)

        keypoints = keypoints.reshape((2,68)).T
        # display(keypoints)

        joint_min_x = int(round(np.nanmin(keypoints[:,0])))
        joint_min_y = int(round(np.nanmin(keypoints[:,1])))

        joint_max_x = int(round(np.nanmax(keypoints[:,0])))
        joint_max_y = int(round(np.nanmax(keypoints[:,1])))

        # print(joint_max_x)

        expand_x = int(round(10/100 * (joint_max_x-joint_min_x)))
        expand_y = int(round(10/100 * (joint_max_y-joint_min_y)))

        if format == "cv2":
            return image[max(0,joint_min_y-expand_y):min(joint_max_y+expand_y, image.shape[0]), max(0,joint_min_x-expand_x):min(joint_max_x+expand_x,image.shape[1])]
        elif format == "PIL":
            bottom = min(joint_max_y+expand_y, image.height)
            right = min(joint_max_x+expand_x,image.width)
            top = max(0,joint_min_y-expand_y)
            left = max(0,joint_min_x-expand_x)
            # print(top, left, bottom, right)
            return tF.crop(image, top, left, bottom-top ,right-left)

    def keypoints(self, index):
        sample = self.df.iloc[index]

        openface_csv = os.path.join(self.db_path, "AllVideos", sample["video"]+"_openface", sample["video"].replace("mp4","csv"))

        df = pd.read_csv(openface_csv)
        df = df.rename(columns=lambda x: x.strip())

        return df[landmarks]

    def _load_image(self, directory, idx, index, mode="body"):
        # print(self.keypoints(index).shape[0] - idx)
        keypoints = self.keypoints(index)
        # print(keypoints.shape, idx)

        if idx >= keypoints.shape[0]:
            idx = keypoints.shape[0]-1

        try:
            keypoints = keypoints.iloc[idx]
        except IndexError as e:
            print("aa", keypoints.shape, idx)
            raise


        sample = self.df.iloc[index]

        if self.modality == 'RGB' or self.modality == 'RGBDiff':

            frame = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert("RGB")

            if keypoints.size == 0:
                face = frame
                pass #just do the whole frame
            elif keypoints.isnull().values.any():
                face = frame
            else:
                # face=frame
                face = self.get_bounding_box(frame, keypoints, format="PIL")

                if face.size == 0:
                    print(keypoints)
                    face = frame

            return [face]

        elif self.modality == 'Flow':
            frame_x = Image.open(os.path.join(directory, self.image_tmpl.format('flow_x', idx))).convert('L')
            frame_y = Image.open(os.path.join(directory, self.image_tmpl.format('flow_y', idx))).convert('L')
       

            if keypoints.size == 0:
                body_x = frame_x
                body_y = frame_y
                pass #just do the whole frame
            elif keypoints.isnull().values.any():
                body_x = frame_x
                body_y = frame_y
            else:
                body_x = self.get_bounding_box(frame_x, keypoints, format="PIL")
                body_y = self.get_bounding_box(frame_y, keypoints, format="PIL")

                if body_x.size == 0:
                    body_x = frame_x
                    body_y = frame_y


            return [body_x, body_y]


    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) # + (record.min_frame+1)
            # print(record.num_frames, record.min_frame, record.max_frame)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        sample = self.df.iloc[index]

        fname = os.path.join(self.db_path,"AllVideos",self.df.iloc[index]["video"])
        # print(fname)

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
        # print(frame_count)
        capture.release()

      
        record_path = os.path.join(self.db_path,"AllFrames",sample["video"].replace(".mp4",""))

        record = VideoRecord([record_path, frame_count])

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        # segment_indices = [100]
        return self.get(record, segment_indices, index)

    def get(self, record, indices, index):

        images = list()
        # print(indices)
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p, index, mode="face")

                images.extend(seg_imgs)

                if p < record.num_frames:
                    p += 1


        categorical = self.df.iloc[index][self.categorical_emotions]
    
        continuous = self.df.iloc[index][self.continuous_emotions]
        continuous = continuous/7.0 # normalize to 0 - 1

        if self.transform is None:
            process_data = images
        else:
            process_data = self.transform(images)


        # ------ AUDIO ------- #
        sample = self.df.iloc[index]

        fname = os.path.join(self.db_path,"AllVideos",self.df.iloc[index]["video"])

        spec = np.load(fname.replace(".mp4","_full.npy"))

        spec = torch.FloatTensor(spec)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
        
        ])

        if transform:
            spec = transform(spec)



        return process_data, torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"], spec
   

    def __len__(self):
        return len(self.df)


import librosa
import torchaudio
import matplotlib.pyplot as plt


class AudioDataSet(data.Dataset):
    def __init__(self, mode, transform=None):

        self.transform = transform

        self.db_path = "/gpu-data/filby/EmoReact_V_1.0/Data"

        self.categorical_emotions = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]

        self.continuous_emotions = ["Valence"]
        
        self.df = pd.read_csv(os.path.join("EmoReact/{}.csv".format(mode)))

        self.video_list = self.df["video"]
        self.mode = mode


    def __getitem__(self, index):

        sample = self.df.iloc[index]

        fname = os.path.join(self.db_path,"AllVideos",self.df.iloc[index]["video"])


        spec = np.load(fname.replace(".mp4","_full.npy"))



        spec = torch.FloatTensor(spec)
        if self.transform:
            spec = self.transform(spec)


        categorical = self.df.iloc[index][self.categorical_emotions]

        continuous = self.df.iloc[index][self.continuous_emotions]
        continuous = continuous/7.0 # normalize to 0 - 1





        return spec, torch.tensor(categorical).float(), torch.tensor(continuous).float(), self.df.iloc[index]["video"], 0


    def __len__(self):
        return len(self.df)
