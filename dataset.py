from torch.utils.data import Dataset #type:ignore
import torch #type:ignore
from torchvision import transforms #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from PIL import Image #type:ignore
from utils import Image_Utils, Data_Utils
import cv2 #type:ignore
import numpy as np #type:ignore
from tqdm import tqdm #type:ignore
import os 
import random


class CustomDataset(Dataset):
    
    def __init__(self, data_ = [], labels_ = [], data_dir= "", ch=4, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = data_
        self.labels = labels_
        self.ch = ch
        if self.data == []:
            self._load_data()

    def _load_data(self):
        for folder in os.listdir(self.data_dir):
            label = 1 if folder == "Positive" else 0
            folder_path = os.path.join(self.data_dir, folder)
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                self.data.append(image_path)
                self.labels.append(label)
        temp = list(zip(self.data, self.labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        if self.ch==3:
            r, g, b, _ = img.split()
            img = Image.merge("RGB", (r, g, b))
        elif self.ch==1:
            _, _, _, a = img.split()
            img = a
        else : pass
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def split(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

        train_data, test_data, train_labels, test_labels = train_test_split(self.data, self.labels, test_size=0.2, random_state=42)

        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

        train_dataset = CustomDataset(data_ = train_data, labels_ = train_labels, transform=train_transform, ch=self.ch)
        val_dataset = CustomDataset(data_ = val_data, labels_ = val_labels, transform=transform, ch=self.ch)
        test_dataset = CustomDataset(data_ = test_data, labels_ = test_labels, transform=transform, ch=self.ch)

        print(f"train dataset : {len(train_dataset)}, val : {len(val_dataset)}, test : {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset, train_labels
    
    def split2(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=10),
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

        train_data, val_data, train_labels, val_labels = train_test_split(
            self.data, self.labels, test_size=0.2, random_state=42
        )


        train_dataset = CustomDataset(data_ = train_data, labels_ = train_labels, transform=train_transform, ch=self.ch)
        val_dataset = CustomDataset(data_ = val_data, labels_ = val_labels, transform=transform, ch=self.ch)

        print(f"train dataset : {len(train_dataset)}, val : {len(val_dataset)}")

        return train_dataset, val_dataset, train_labels
    
    def get_pratio(self):
        """
        Get the ratio of positive samples in the dataset

        Parameters
        ----
        None

        Returns
        ----
        float: number of positive samples
        float: number of negative samples
        """
        num_positive = sum(label == 1 for _, label in self)
        num_negative = sum(label == 0 for _, label in self)
        print(f"Number of positive samples: {num_positive}")
        print(f"Number of negative samples: {num_negative}")

        return num_positive, num_negative
    
    
    def get_weight(self, device="cpu"):
        """
        Get the class weights for the dataset for use in the weighted cross entropy loss function

        Parameters
        ----
        device: str
            The device to move the class weights to

        Returns
        ----
        torch.Tensor: The class weights
        """
        class_counts = torch.bincount(torch.tensor(self.labels))
        total_samples = len(self.labels)
        class_frequencies = class_counts.float() / total_samples
        class_weights = 1.0 / class_frequencies
        class_weights /= class_weights.sum()
        class_weights = class_weights.to(device)

        return class_weights
    
    @staticmethod
    def square_save(image, output_folder = "temp/temp/", stride = 24):
        """
        Save the image cut into squares

        Parameters
        ----
        image: numpy.ndarray
            The image to save the squares from
        output_folder: str
            The folder to save the squares in
        stride: int
            The stride to use when saving the squares (default 24)

        Returns
        ----
        None
        """

        Data_Utils.create_folder(output_folder, rm=True)
        image_width, image_height, _ = image.shape
        print(image.shape)
        square_size = 96

        num_rows = (image_width - square_size) // stride + 1
        num_cols = (image_height - square_size) // stride + 1

        for row in range(num_rows):
            for col in range(num_cols):
                left = col * stride
                top = row * stride
                right = left + square_size
                bottom = top + square_size

                square = image[top:bottom, left:right]

                filename = f"square_{row}_{col}.png"
                output_path = f"{output_folder}{filename}"
                Image_Utils.save(output_path, square)

    @staticmethod
    def create(img_path = "../Images/vn_train1.png", crack_img_path= "../Images/vn_train1_crack.png", dataset_path = "Dataset/", rm=False):
        """
        Create the dataset from the image and the crack image

        Parameters
        ----
        img_path: str
            The path to the image
        crack_img_path: str
            The path with the annotated cracks
        dataset_path: str   
            The path to the dataset folder
        rm: bool
            Whether to remove the dataset folder if it already exists (default False)

        Returns
        ----
        None
        """

        Data_Utils.create_folder(dataset_path+"Negative/", rm=rm)
        Data_Utils.create_folder(dataset_path+"Positive/", rm=rm)
        Data_Utils.create_folder("temp/temp/", rm=True)
        Data_Utils.create_folder("temp/tempcrack/", rm=True)
        
        image = cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
        CustomDataset.square_save(image)
        image_crack = Image_Utils.load(crack_img_path)
        CustomDataset.square_save(image_crack, output_folder="temp/tempcrack/")
        print("done")

        chemins = Data_Utils.find_path("temp/temp")
        chemins_crack = Data_Utils.find_path("temp/tempcrack")

        nb_pos = 0
        
        for chemin, chemin_crack in zip(tqdm(chemins), chemins_crack):

            square = cv2.imread(chemin,cv2.IMREAD_UNCHANGED)
            name = os.path.basename(chemin)
            avg = np.mean(Image_Utils.load(chemin_crack))

            if avg >= 254 : test = 0
            else : test=1

            if int(test)==1:
                Image_Utils.save(dataset_path+"Positive/"+name, square)
                nb_pos += 1
            elif int(test)==123: break
            else:
                Image_Utils.save(dataset_path+"Negative/"+name, square)

        print(f"{nb_pos} positives found")
        

if __name__ == "__main__":
    
    CustomDataset.create(
        img_path = "../Images/vn_train.png", 
        crack_img_path= "../Images/vn_crack.png", 
        dataset_path = "Dataset/",
        rm=True
    )