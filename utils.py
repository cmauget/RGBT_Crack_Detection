import matplotlib.pyplot as plt #type: ignore
import cv2 #type: ignore
import os
import numpy as np #type: ignore
from PIL import Image #type: ignore

#---------------Image_Utils----------------#

class Image_Utils:

    @staticmethod
    def save(image_path: str, image) -> None:
        """Save an image using opencv.

        Parameters
        ----
        image_path : str
            The path to save the image.
        image : numpy.ndarray
            The image to be saved.
        """
        cv2.imwrite(image_path, image)

    @staticmethod
    def load(image_path: str):
        """Load a standard image
        
        Parameters
        ----
        image_path : str
            The path of the image to load

        Returns
        ----
        img : numpy.ndarray
            The loaded image
        """
        if os.path.basename(image_path) == ".DS_Store":
            print("File .DS_Store found, please restart the program")
            img = None
            os.remove(image_path)
        else:
            img = cv2.imread(image_path)
        return img

    @staticmethod
    def load_streamlit(uploaded_file, image_path="temp_image.png", bw=False):
        """Load an image from the Streamlit file uploader widget
        
        Parameters
        ----
        uploaded_file : 
            Output of the file uploader widget
        image_path : str, optional
            Temporary file where the image will be saved
        bw : boolean, optional
            If true, load the image in black and white
        
        Returns
        ----
        img : numpy.ndarray
            The loaded image
        """
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if bw:
            img = cv2.imread(image_path, 0)
        else: 
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
        os.remove(image_path)
        return img

    @staticmethod
    def invert_image(image):
        """Invert an image
        
        Parameters
        ----
        image : numpy.ndarray
            The image to invert
        
        Returns
        ----
        img : numpy.ndarray
            The inverted image
        """
        img = cv2.bitwise_not(image)
        return img

    @staticmethod
    def load_streamlit_PIL(uploaded_file, image_path="temp_image.png", bw=False):
        """Load and save an image from the Streamlit file uploader widget using PIL.

        Parameters:
        uploaded_file : BytesIO
            The output of the file uploader widget.
        image_path : str, optional
            The temporary file path where the image will be saved.
        bw : bool, optional
            If True, the image will be loaded in black and white (grayscale).

        Returns:
        img : numpy.ndarray
            The loaded image as a NumPy array.
        """
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        img = Image.open(image_path)

        if bw:
            img = img.convert("L")

        os.remove(image_path)

        return img

    @staticmethod
    def resize_img(image_hr, ratio=4):
        """Resize the image for super-resolution

        1/ Create an image with dimensions divisible by 4
        2/ Reduce the image resolution by four
        
        Parameters
        ----
        image : numpy.ndarray
            The image to resize
        ratio : int, optional
            The ratio to reduce the image
        
        Returns
        ----
        image_hr2: numpy.ndarray
            The image divisible by 4
        image_lr: numpy.ndarray
            The resized image
        """
        largeur_hr = image_hr.shape[1]
        hauteur_hr = image_hr.shape[0]

        largeur_hr2 = (largeur_hr // ratio) * ratio
        hauteur_hr2 = (hauteur_hr // ratio) * ratio

        image_hr2 = cv2.resize(image_hr, (largeur_hr2, hauteur_hr2))
        
        largeur_lr = largeur_hr2 // ratio
        hauteur_lr = hauteur_hr2 // ratio
        image_lr = cv2.resize(image_hr2, (largeur_lr, hauteur_lr))

        return image_hr2, image_lr

    @staticmethod
    def crop_img(image, crop_coords):
        """Crop the image based on the input coordinates
        
        Parameters
        ----
        image : numpy.ndarray
            The image to crop
        croop_coords : tuples
            The coordinates to crop in the order [x1, x2, y1, y2]
        
        Returns
        ----
        img: numpy.ndarray
            The cropped image
        """

        img = image[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        return img

#-------------Data_Utils--------------#

class Data_Utils:

    @staticmethod
    def graphe(liste_image, liste_titre) -> None:
        """Display images in rows of 3

        The first two titles are displayed in bold
        
        Parameters
        ----
        liste_image : list[numpy.ndarray]
            List of images to display
        liste_titre : list[str]
            List of titles associated with the images
        """
        size = len(liste_image)
        if size % 3 == 0:
            height = size // 3
        else:
            height = len(liste_image) // 3 + 1
        for i, (image, titre) in enumerate(zip(liste_image, liste_titre), start=1):
                dec = 40
                plt.subplot(height, 3, i)
                if i == 1:
                    dec = dec / 4
                    plt.title(titre, fontweight='bold')
                elif i == 2:
                    plt.title(titre, fontweight='bold')
                else:
                    plt.title(titre)
                # Original image
                plt.imshow(image[:, :, ::-1], origin="lower")

                plt.xlim(image.shape[1] // 8 + dec, 2 * image.shape[1] // 8 + dec)  
                plt.ylim(2 * image.shape[0] // 8, image.shape[0] // 8) 
                plt.axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def find_path(folder_path: str) -> list[str]:
        """Find paths of images in a folder
        
        Parameters
        ----
        folder_path : str
            The folder where the images are located

        Returns
        ----
        liste_chemin: list[str]
            The list of paths of all the images
        """
        liste_chemin = []
        for nom_fichier in sorted(os.listdir((folder_path))):
            chemin_ = os.path.join(folder_path, nom_fichier)
            liste_chemin.append(chemin_)
        
        return liste_chemin

    @staticmethod
    def create_folder(folder_path: str, rm=False) -> None:
        """Create a folder

        It checks if the folder exists, creates it if not, and deletes
        its contents before if needed
        
        Parameters
        ----
        folder_path : str
            The path of the folder to create
        rm : Boolean, optional
            If true, the contents of the folder will be deleted before
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if rm:
            for chemin_fichier in Data_Utils.find_path(folder_path): 
                if os                .path.isfile(chemin_fichier):  
                    os.remove(chemin_fichier)  

    @staticmethod
    def resize(dossier_hr: str, dossier_hr2: str, dossier_lr: str, ratio=4) -> None:
        """Resize the image for super-resolution (legacy)

        Takes the paths of the folders as input
        """
        Data_Utils.create_folder(dossier_hr2)
        Data_Utils.create_folder(dossier_lr)

        for nom_fichier in os.listdir(dossier_hr):
            chemin_hr = os.path.join(dossier_hr, nom_fichier)
            
            image_hr = cv2.imread(chemin_hr)

            largeur_hr = image_hr.shape[1]
            hauteur_hr = image_hr.shape[0]

            largeur_hr2 = (largeur_hr // ratio) * ratio
            hauteur_hr2 = (hauteur_hr // ratio) * ratio

            image_hr2 = cv2.resize(image_hr, (largeur_hr2, hauteur_hr2))
            
            chemin_hr2 = os.path.join(dossier_hr2, nom_fichier)
            
            cv2.imwrite(chemin_hr2, image_hr2)
            
            largeur_lr = largeur_hr2 // ratio
            hauteur_lr = hauteur_hr2 // ratio
            image_lr = cv2.resize(image_hr2, (largeur_lr, hauteur_lr))
            
            chemin_lr = os.path.join(dossier_lr, nom_fichier)
            
            cv2.imwrite(chemin_lr, image_lr)

        print("Image resolution reduction is complete.")

    @staticmethod    
    def crop(input_image: str, output_image: str, crop_coords) -> None:
        """Crop the image (legacy)

        Takes the paths of the folders as input
        """
        img = cv2.imread(input_image)
        cropped_img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        cv2.imwrite(output_image, cropped_img)

    @staticmethod
    def convert(dossier_ir: str, dossier_hr: str) -> None:
        """Convert the 1-channel image to RGB (legacy)

        Takes the paths of the folders as input
        """
        chemin_ir = []
        Data_Utils.create_folder(dossier_hr)
        chemin_ir = Data_Utils.find_path(dossier_ir)

        for chemin in chemin_ir:

            image = Data_Utils.loadio(chemin)

            image_scaled = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            image_rgb = cv2.cvtColor(image_scaled, cv2.COLOR_GRAY2RGB)

            nom_fichier_ = os.path.basename(chemin)
            chemin_ = os.path.join(dossier_hr, nom_fichier_)
            cv2.imwrite(chemin_, image_rgb)


class Metrics_Utils:

    @staticmethod
    def calculate_recall(conf_matrix):
        """Calculate recall from the confusion matrix
        
        Parameters
        ----
        conf_matrix : [2x2] confusion matrix
        """
        true_positives = conf_matrix[1, 1]
        false_negatives = conf_matrix[1, 0]
        if true_positives + false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        return recall

    @staticmethod
    def calculate_precision(conf_matrix):
        """Calculate precision from the confusion matrix
        
        Parameters
        ----
        conf_matrix : [2x2] confusion matrix
        """
        true_positives = conf_matrix[1, 1]
        false_positives = conf_matrix[0, 1]
        if true_positives + false_positives == 0:
            precision = 0
        else:
            precision = true_positives / (true_positives + false_positives)
        return precision

    @staticmethod
    def calculate_f1_score(conf_matrix):
        """Calculate f1-score from the confusion matrix
        
        Parameters
        ----
        conf_matrix : [2x2] confusion matrix
        """
        recall = Metrics_Utils.calculate_recall(conf_matrix)
        precision = Metrics_Utils.calculate_precision(conf_matrix)
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    @staticmethod
    def calculate_accuracy_(conf_matrix):
        """Calculate accuracy from the confusion matrix
        
        Parameters
        ----
        conf_matrix : [2x2] confusion matrix
        """
        true = conf_matrix[1, 1] + conf_matrix[0, 0]
        false = conf_matrix[1, 0] + conf_matrix[0, 1]
        accuracy = true / (true + false)
        return accuracy
