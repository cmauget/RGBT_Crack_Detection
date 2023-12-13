import torch #type:ignore
import torch.nn as nn #type:ignore
from torch.utils.data import DataLoader #type:ignore
from dataset import CustomDataset
import torch.optim as optim #type:ignore
from sklearn.metrics import confusion_matrix #type:ignore
import matplotlib.pyplot as plt #type:ignore
from tqdm import tqdm #type:ignore
from model import CNNModel, FocalLoss
from utils import Metrics_Utils as m
from torchvision import transforms #type:ignore
import os #type:ignore

BATCH_SIZE = 64
NUM_EPOCHS = 70
DEVICE = "mps"
CH = 3
DATA_DIR = "Dataset"
LOSS = "focalloss"
wandb = True #wandb is a logging tool, if you don't want to use it, set it to False

if wandb:
    import wandb#type:ignore
    wandb.login()

    run = wandb.init(
                        project="RGBT_Crack_Detetion",
                        config={"ch": CH,
                                "batch_size": BATCH_SIZE,
                                "num_epochs": NUM_EPOCHS,
                                "loss": LOSS,
                        }
                    )
    
    results_dir = './run-' + run.name

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
else:
    results_dir = './run'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)



def calculate_loss(model, data_loader, criterion):
    """
    Calculates the loss of the model on data from the data_loader
    
    Parameters
    ----
    model: torch.nn.Module
        The model to use for prediction
    data_loader: torch.utils.data.DataLoaderr
        The data to use for prediction
    criterion: torch.nn.Module
        The loss function to use for prediction
        
    Returns
    ----
    loss: float
        The loss of the model on the data
    """
    model.eval()
    running_loss = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.long())
            running_loss += loss.item()
    return (running_loss/len(data_loader))


def main(num_epochs = NUM_EPOCHS, device = DEVICE, ch = CH, data_dir = DATA_DIR, loss = LOSS, batch_size = BATCH_SIZE):
    """
    Train the model relatively to the given parameters

    Parameters
    ----
    num_epochs: int
        The number of epochs to train the model
    device: str
        The device to use for training (cpu, cuda or mps (rocm one day?))
    ch: int
        The number of channels to use for training (3 for RGB, 4 for RGBT)
    data_dir: str
        The directory where the dataset is stored
    loss: str
        The loss function to use for training (either focalloss or crossentropy)
    batch_size: int
        The batch size to use for training (reduce if out of memory issues)
    """

    dataset = CustomDataset(data_dir = data_dir, ch=CH)
    train_dataset, val_dataset, test_dataset, _ = dataset.split()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = CNNModel(device_=device, ch=ch)

    class_weights = train_dataset.get_weight(device=DEVICE)
    print(class_weights)

    if LOSS == "focalloss":
        criterion =  FocalLoss(alpha=1, gamma=2, reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight = class_weights)
    #criterion =  FocalLoss(alpha=1, gamma=2, reduction='sum') #nn.CrossEntropyLoss(weight = class_weights) #FocalLoss(alpha=1, gamma=2, reduction='sum') # weight = class_weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    vloss = []
    tloss = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        total=0
        true_labels = []
        predicted_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
        
        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        f1 = m.calculate_f1_score(conf_matrix)
        train_accuracy = m.calculate_accuracy_(conf_matrix)
        val_loss = calculate_loss(model, val_loader, criterion=criterion)
        
        vloss.append(val_loss)
        tloss.append(running_loss/len(train_loader))

        if wandb:
            wandb.log({"Train Loss": running_loss/len(train_loader), "Val Loss": val_loss, "F1": f1, "Accuracy": train_accuracy, "Epoch": epoch+1})
        else:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss/len(train_loader):.10f}, F1: {f1:.3f}, Acc: {train_accuracy:.3f}, Val Loss {val_loss:.3f}")

        torch.save(model.state_dict(), f'{results_dir}/model_{CH}.pt')

    wandb.finish()


    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    recall = m.calculate_recall(conf_matrix)
    f1 = m.calculate_f1_score(conf_matrix)
    precision = m.calculate_precision(conf_matrix)
    accuracy = m.calculate_accuracy_(conf_matrix)
    output_str = f"Confusion Matrix: {conf_matrix}\nF1 score: {f1}, Recall: {recall}, Precision: {precision}, Acc: {accuracy}"
    file_path = f'{results_dir}/test_metrics.txt'
    with open(file_path, 'w') as file:
        file.write(output_str)

if __name__ == "__main__":
    main(num_epochs = NUM_EPOCHS, device = DEVICE, ch = CH, data_dir = DATA_DIR, loss = LOSS, batch_size=BATCH_SIZE)