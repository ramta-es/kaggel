from torch.optim import Adam
import torch.nn as nn
import time as time
from datasets_objects.datasets import MriDataset
from models.segmentation_models import UNet

import torch

im_root = 'dataset/uw-madison-gi-tract-image-segmentation'
csv_path = '/dataset/uw-madison-gi-tract-image-segmentation/train.csv'
csv_path = '/dataset/uw-madison-gi-tract-image-segmentation/train.csv'
config_path = '/torch_trainer/example_config_files/unet_model.yaml'


class Trainer():
    def __init__(self, model, data_object, criterion, optimizer, **kwargs):
        self.model = model
        self.criterion = criterion
        self.optim = optimizer
        # self.cfg = cfg # TODO: change model to configuration
        self.data = data_object#(im_root, csv_path)  # TODO: change to general form
        self.train_loader, self.test_loader = self.data.get_dataloaders(batch_size=2)
        self.train_loss = []
        self.test_loss = []
        self.train_correct = []
        self.test_correct = []

    def train(self):
        global device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        start_time = time.time()
        total_time = time.time() - start_time
        for epoch in range(2):
            model.train()
            tr_cor, tr_loss = self.train_loop(train_loader=self.train_loader, criterion=self.criterion, optimizer=self.optim, epoch=epoch)
            self.train_loss.append(tr_loss)
            self.train_correct.append(tr_cor)
            tst_corr, tst_loss = self.eval_loop(test_loader=self.test_loader, epoch=epoch, criterion=self.criterion)
            self.test_loss.append(tst_loss)
            self.test_correct.append(tst_corr)

        print(f'Total time: {total_time / 60} minutes')

    @staticmethod
    def train_loop(train_loader, epoch, criterion, optimizer, trn_corre=0):
        for i, (X_train, y_train) in enumerate(train_loader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            y_pred = model(X_train.float())
            loss = criterion(y_pred, y_train.squeeze(1).long())
            print('loss', loss)

            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum().item()
            trn_corre += batch_corr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch: {epoch} Loss train: {loss.item()}')

        return trn_corre, loss

    @staticmethod
    def eval_loop(test_loader, epoch, criterion, tst_corr=0):
        with torch.no_grad():
            for i, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.to(device)
                y_test = y_test.to(device)

                y_val = model(X_test.float())
                predicted = torch.max(y_val.data, 1)[1]
                batch_corr = (predicted == y_test).sum().item()
                tst_corr += batch_corr
        loss = criterion(y_val, y_test.squeeze(1).long())
        print(f'Epoch: {epoch} Loss test: {loss.item()}')
        return tst_corr, loss


model = UNet(n_channels=1, n_classes=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = MriDataset(im_root=im_root, csv_path=csv_path)
train_loader, test_loader = data.get_dataloaders(batch_size=2)

epochs = range(2)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)



t = Trainer(model=model, data_object=data, criterion=criterion, optimizer=optimizer)
t.train()




