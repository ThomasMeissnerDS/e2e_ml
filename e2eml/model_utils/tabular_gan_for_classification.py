import gc
import logging
import os
import re

# import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim

# from IPython.display import clear_output
# from sklearn.metrics import matthews_corrcoef
from torch.autograd.variable import Variable
from torch.utils.data import DataLoader, Dataset

from e2eml.full_processing.postprocessing import FullPipeline

# from tqdm import tqdm
# from transformers import AdamW, get_linear_schedule_with_warmup


# specify GPU
scaler = torch.cuda.amp.GradScaler()  # GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GanDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TabularGeneratorClassification(FullPipeline, GanDataset):
    def get_generator_discriminator(self):
        class Generator(nn.Module):
            """
            Creates fake data trying to trick the Discriminator.
            """

            def __init__(self, num_features, output_dim, dropout=0.3):
                super(Generator, self).__init__()

                self.dropout = dropout

                self.layer_0 = nn.Linear(num_features, 4096)
                self.batch_norm_0 = nn.BatchNorm1d(4096)

                self.layer_1 = nn.Linear(4096, 64)
                self.batch_norm_1 = nn.BatchNorm1d(64)

                self.layer_2 = nn.Linear(64, 128)
                self.batch_norm_2 = nn.BatchNorm1d(128)

                self.layer_3 = nn.Linear(128, 256)
                self.batch_norm_3 = nn.BatchNorm1d(256)

                self.layer_4 = nn.Linear(256, 512)
                self.batch_norm_4 = nn.BatchNorm1d(512)

                self.layer_5 = nn.Linear(512, 16)
                self.batch_norm_5 = nn.BatchNorm1d(16)

                self.layer_6 = nn.Linear(16, 512)
                self.batch_norm_6 = nn.BatchNorm1d(512)

                self.layer_7 = nn.Linear(512, 256)
                self.batch_norm_7 = nn.BatchNorm1d(256)

                self.layer_8 = nn.Linear(256, 128)
                self.batch_norm_8 = nn.BatchNorm1d(128)

                self.layer_9 = nn.Linear(128, 16)
                self.batch_norm_9 = nn.BatchNorm1d(16)

                self.layer_out = nn.Linear(16, output_dim)

                self.silu = nn.SiLU()

            def forward(self, inputs):
                x = self.silu(self.layer_0(inputs))
                x = self.batch_norm_0(x)
                x = self.silu(self.layer_1(x))
                x = self.batch_norm_1(x)
                x = self.silu(self.layer_2(x))
                x = self.batch_norm_2(x)
                x = self.silu(self.layer_3(x))
                x = self.batch_norm_3(x)
                x = self.silu(self.layer_4(x))
                x = self.batch_norm_4(x)
                x = self.silu(self.layer_5(x))
                x = self.batch_norm_5(x)
                x = self.silu(self.layer_6(x))
                x = self.batch_norm_6(x)
                x = self.silu(self.layer_7(x))
                x = self.batch_norm_7(x)
                x = self.silu(self.layer_8(x))
                x = self.batch_norm_8(x)
                x = self.silu(self.layer_9(x))
                x = self.batch_norm_9(x)
                x = self.layer_out(x)
                return x

            def predict(self, inputs):
                x = self.silu(self.layer_0(inputs))
                x = self.batch_norm_0(x)
                x = self.silu(self.layer_1(x))
                x = self.batch_norm_1(x)
                x = self.silu(self.layer_2(x))
                x = self.batch_norm_2(x)
                x = self.silu(self.layer_3(x))
                x = self.batch_norm_3(x)
                x = self.silu(self.layer_4(x))
                x = self.batch_norm_4(x)
                x = self.silu(self.layer_5(x))
                x = self.batch_norm_5(x)
                x = self.silu(self.layer_6(x))
                x = self.batch_norm_6(x)
                x = self.silu(self.layer_7(x))
                x = self.batch_norm_7(x)
                x = self.silu(self.layer_8(x))
                x = self.batch_norm_8(x)
                x = self.silu(self.layer_9(x))
                x = self.batch_norm_9(x)
                x = self.layer_out(x)
                return x

        class Discriminator(nn.Module):
            """
            Will decide, if data is real or fake.
            """

            def __init__(self, num_features, dropout=0.3):
                super(Discriminator, self).__init__()

                self.dropout = dropout

                self.layer_0 = nn.Linear(num_features, 4096)
                self.batch_norm_0 = nn.BatchNorm1d(4096)
                self.dropout_0 = nn.Dropout(dropout)

                self.layer_1 = nn.Linear(4096, 64)
                self.batch_norm_1 = nn.BatchNorm1d(64)
                self.dropout_1 = nn.Dropout(dropout)

                self.layer_2 = nn.Linear(64, 128)
                self.batch_norm_2 = nn.BatchNorm1d(128)

                self.layer_3 = nn.Linear(128, 256)
                self.batch_norm_3 = nn.BatchNorm1d(256)

                self.layer_4 = nn.Linear(256, 512)
                self.batch_norm_4 = nn.BatchNorm1d(512)
                self.dropout_4 = nn.Dropout(dropout)

                self.layer_5 = nn.Linear(512, 16)
                self.batch_norm_5 = nn.BatchNorm1d(16)

                self.layer_6 = nn.Linear(16, 512)
                self.batch_norm_6 = nn.BatchNorm1d(512)

                self.layer_7 = nn.Linear(512, 256)
                self.dropout_7 = nn.Dropout(dropout)
                self.batch_norm_7 = nn.BatchNorm1d(256)

                self.layer_8 = nn.Linear(256, 128)
                self.batch_norm_8 = nn.BatchNorm1d(128)

                self.layer_9 = nn.Linear(128, 16)
                self.batch_norm_9 = nn.BatchNorm1d(16)

                self.layer_out = nn.Linear(16, 1)

                self.silu = nn.SiLU()

                self.sigmoid = nn.Sigmoid()

            def forward(self, inputs):
                x = self.silu(self.layer_0(inputs))
                x = self.batch_norm_0(x)
                x = self.dropout_0(x)
                x = self.silu(self.layer_1(x))
                x = self.batch_norm_1(x)
                x = self.dropout_1(x)
                x = self.silu(self.layer_2(x))
                x = self.batch_norm_2(x)
                x = self.silu(self.layer_3(x))
                x = self.batch_norm_3(x)
                x = self.silu(self.layer_4(x))
                x = self.batch_norm_4(x)
                x = self.dropout_4(x)
                x = self.silu(self.layer_5(x))
                x = self.batch_norm_5(x)
                x = self.silu(self.layer_6(x))
                x = self.batch_norm_6(x)
                x = self.silu(self.layer_7(x))
                x = self.dropout_7(x)
                x = self.batch_norm_7(x)
                x = self.silu(self.layer_8(x))
                x = self.batch_norm_8(x)
                x = self.silu(self.layer_9(x))
                x = self.batch_norm_9(x)
                x = self.layer_out(x)
                # x = self.silu(self.layer_out(x))
                # x = self.sigmoid(x)
                return x

            def predict(self, inputs):
                x = self.silu(self.layer_0(inputs))
                x = self.batch_norm_0(x)
                x = self.dropout_0(x)
                x = self.silu(self.layer_1(x))
                x = self.batch_norm_1(x)
                x = self.dropout_1(x)
                x = self.silu(self.layer_2(x))
                x = self.batch_norm_2(x)
                x = self.silu(self.layer_3(x))
                x = self.batch_norm_3(x)
                x = self.silu(self.layer_4(x))
                x = self.batch_norm_4(x)
                x = self.dropout_4(x)
                x = self.silu(self.layer_5(x))
                x = self.batch_norm_5(x)
                x = self.silu(self.layer_6(x))
                x = self.batch_norm_6(x)
                x = self.silu(self.layer_7(x))
                x = self.dropout_7(x)
                x = self.batch_norm_7(x)
                x = self.silu(self.layer_8(x))
                x = self.batch_norm_8(x)
                x = self.silu(self.layer_9(x))
                x = self.batch_norm_9(x)
                x = self.layer_out(x)
                # x = self.silu(self.layer_out(x))
                # x = self.sigmoid(x)
                return x

        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        n_features = X_train.values.shape[1]
        generator = Generator(num_features=n_features, output_dim=n_features)
        discriminator = Discriminator(num_features=n_features)

        generator.to(device)
        discriminator.to(device)
        return generator, discriminator

    def noise(self, n=None):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        n_features = X_train.values.shape[1]
        if not n:
            n = X_train.values.shape[0]
        return Variable(torch.randn(n, n_features)).to(device)

    def make_ones(self, size):
        data = Variable(torch.ones(size, 1))
        return data.to(device)

    def make_zeros(self, size):
        data = Variable(torch.zeros(size, 1))
        return data.to(device)

    def gan_model_setup(self):
        generator, discriminator = self.get_generator_discriminator()

        g_optim = optim.RMSprop(
            generator.parameters(),
            lr=self.gan_settings["generator_learning_rate"],
        )
        d_optim = optim.RMSprop(
            generator.parameters(),
            lr=self.gan_settings["discriminator_learning_rate"],
        )

        loss_fn = nn.BCELoss()
        return generator, discriminator, g_optim, d_optim, loss_fn

    def train_discriminator(
        self, optimizer, real_data, fake_data, discriminator, loss_fn
    ):
        optimizer.zero_grad()

        prediction_real = discriminator(real_data)

        prediction_fake = discriminator(fake_data)

        D_loss = -(torch.mean(prediction_real) - torch.mean(prediction_fake))

        D_loss.backward()
        optimizer.step()

        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        return D_loss

    def train_generator(self, optimizer, fake_data, discriminator, loss_fn):
        optimizer.zero_grad()

        prediction = discriminator(fake_data)

        G_loss = -torch.mean(prediction)

        G_loss.backward()
        optimizer.step()

        return G_loss, optimizer

    def create_gan_train_dataset(self, target_class):
        logging.info("Create NLP train dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

        X_train_class_only = X_train.iloc[np.where(Y_train == target_class)[0]]
        Y_train_class_only = Y_train.iloc[np.where(Y_train == target_class)[0]]

        train_dataset = GanDataset(
            torch.from_numpy(X_train_class_only.to_numpy().astype(float)).float(),
            torch.from_numpy(Y_train_class_only.to_numpy().astype(float)).float(),
        )
        return train_dataset

    def create_gan_train_dataloader(
        self, target_class, train_batch_size=None, workers=None
    ):
        if train_batch_size:
            pass
        else:
            train_batch_size = self.gan_settings["batch_size"]

        if workers:
            pass
        else:
            workers = self.gan_settings["num_workers"]
        logging.info("Create Neural network train dataloader.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        train_dataset = self.create_gan_train_dataset(target_class=target_class)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        return train_dataloader

    def train_gan(self, target_class):
        epochs = self.gan_settings["max_epochs"]
        k = self.gan_settings["discriminator_extra_training_rounds"]
        best_epoch = -1
        # test_noise = self.noise()
        train_dataloader = self.create_gan_train_dataloader(target_class=target_class)
        generator, discriminator, g_optim, d_optim, loss_fn = self.gan_model_setup()

        generator.train()
        discriminator.train()

        g_losses = []
        d_losses = []

        best_loss = 1000000000000
        for epoch in range(epochs):
            g_loss = 0.0
            d_loss = 0.0
            for i, data in enumerate(train_dataloader):
                imgs, _ = data
                n = len(imgs)
                for _j in range(k):
                    fake_data = generator(self.noise(n)).detach()
                    real_data = imgs.to(device)
                    d_loss += self.train_discriminator(
                        d_optim,
                        real_data,
                        fake_data,
                        discriminator=discriminator,
                        loss_fn=loss_fn,
                    )
                fake_data = generator(self.noise(n))
                g_loss_it, optimizer = self.train_generator_regression(
                    g_optim, fake_data, discriminator=discriminator, loss_fn=loss_fn
                )
                g_loss = g_loss + g_loss_it

                del fake_data
                del real_data
                _ = gc.collect()

                # img = generator(test_noise).cpu().detach()
                g_losses.append(g_loss / i)
                d_losses.append(d_loss / i)

            if epoch > best_epoch + self.gan_settings["early_stopping_rounds"]:
                break

            if epoch == 0 or g_loss < best_loss:
                print(
                    f"Found better model in epoch {epoch} with generator loss {g_loss}"
                    f" and discriminator loss {d_loss}."
                )
                best_epoch = epoch
                best_loss = g_loss
                state = {
                    "state_dict": generator.state_dict(),
                    "optimizer_dict": optimizer.state_dict(),
                    "bestscore": g_loss,
                }
                torch.save(state, f"class_{target_class}_generator_model.pth")
                # clear_output()
        del discriminator
        del generator

    def train_class_generators(self):
        self.get_current_timestamp(task="Start training GANs.")

        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        unique_classes = np.unique(Y_train.values)
        for clas in unique_classes:
            self.train_gan(target_class=clas)
        self.get_current_timestamp(task="Finished training GANs.")

    def load_generator_model_states(self, path=None, target_class=None):
        logging.info("Load model save states.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if path:
            pass
        else:
            path = os.getcwd()
        if self.prediction_mode:
            pthes = [
                os.path.join(f"{path}/", s)
                for s in os.listdir(f"{path}/")
                if re.search(f"class_{target_class}_generator_model.pth", s)
            ]
            return pthes
        else:
            pthes = [
                os.path.join(f"{path}/", s)
                for s in os.listdir(f"{path}/")
                if re.search(f"class_{target_class}_generator_model.pth", s)
            ]
            return pthes

    def create_synthetic_data(self):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        columns = X_train.columns
        unique_classes = np.unique(Y_train.values)

        rows_to_create = self.gan_settings["nb_synthetic_rows_to_create"]
        batch_size = self.gan_settings["batch_size"]
        nb_batches = int(rows_to_create / batch_size)
        new_class_data = []
        new_targets = []

        for clas in unique_classes:
            new_data = []
            (
                generator,
                discriminator,
                g_optim,
                d_optim,
                loss_fn,
            ) = self.gan_model_setup()
            pathes = self.load_generator_model_states(
                path=self.gan_model_save_states_path, target_class=clas
            )
            print(pathes)
            for m_path in pathes:
                state = torch.load(m_path)
                generator.load_state_dict(state["state_dict"])
                generator.to(device)
                generator.eval()
                with torch.no_grad():
                    for _batch in range(nb_batches):
                        synth_data = generator.predict(self.noise(batch_size))
                        synth_data = synth_data.cpu().detach()
                        new_data.append(synth_data)

            full_new_data = np.concatenate(new_data, axis=0)
            new_train = pd.DataFrame(full_new_data, columns=columns)
            new_train[self.target_variable] = clas

            new_target = new_train[self.target_variable]
            new_targets.append(new_target.values)

            new_train = new_train.drop(self.target_variable, axis=1)
            new_class_data.append(new_train)

            del generator

        X_train = pd.concat(new_class_data)
        Y_train = pd.Series(np.concatenate(new_targets, axis=0))
        X_train[self.target_variable] = Y_train.values

        X_train = X_train.reset_index(drop=True)
        Y_train = X_train[self.target_variable].copy()
        X_train = X_train.drop(self.target_variable, axis=1)

        try:
            X_test = X_test.drop(self.target_variable, axis=1)
        except KeyError:
            pass

        self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
