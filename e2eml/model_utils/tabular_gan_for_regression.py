import gc
import logging
import os

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
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


class TabularGeneratorRegression(FullPipeline, GanDataset):
    def get_generator_discriminator_regression(self):
        class GeneratorRegression(nn.Module):
            """
            Creates fake data trying to trick the Discriminator.
            """

            def __init__(self, num_features, output_dim, dropout=0.3):
                super(GeneratorRegression, self).__init__()

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

        class DiscriminatorRegression(nn.Module):
            """
            Will decide, if data is real or fake.
            """

            def __init__(self, num_features, dropout=0.3):
                super(DiscriminatorRegression, self).__init__()

                self.dropout = dropout

                self.layer_0 = nn.Linear(num_features, 512)
                self.batch_norm_0 = nn.BatchNorm1d(512)

                self.layer_2 = nn.Linear(512, 16)
                self.batch_norm_2 = nn.BatchNorm1d(16)

                self.layer_3 = nn.Linear(16, 512)
                self.batch_norm_3 = nn.BatchNorm1d(512)

                self.layer_out = nn.Linear(512, 1)

                self.silu = nn.SiLU()

            def forward(self, inputs):
                x = self.silu(self.layer_0(inputs))
                x = self.batch_norm_0(x)
                x = self.silu(self.layer_2(x))
                x = self.batch_norm_2(x)
                x = self.silu(self.layer_3(x))
                x = self.batch_norm_3(x)
                x = self.layer_out(x)
                return x

            def predict(self, inputs):
                x = self.silu(self.layer_0(inputs))
                x = self.batch_norm_0(x)
                x = self.silu(self.layer_2(x))
                x = self.batch_norm_2(x)
                x = self.silu(self.layer_3(x))
                x = self.batch_norm_3(x)
                x = self.layer_out(x)
                return x

        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        X_train[self.target_variable] = Y_train
        n_features = X_train.values.shape[1]
        generator = GeneratorRegression(
            num_features=n_features,
            output_dim=n_features,
            dropout=self.gan_settings["generator_dropout_rate"],
        )
        discriminator = DiscriminatorRegression(
            num_features=n_features,
            dropout=self.gan_settings["discriminator_dropout_rate"],
        )

        generator.to(device)
        discriminator.to(device)
        return generator, discriminator

    def noise(self, n=None):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        X_train[self.target_variable] = Y_train
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

    def gan_model_setup_regression(self):
        generator, discriminator = self.get_generator_discriminator_regression()
        # Initialize weights

        g_optim = optim.Adam(
            generator.parameters(),
            lr=self.gan_settings["generator_learning_rate"],
            betas=(0.5, 0.999),
        )
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=self.gan_settings["discriminator_learning_rate"],
            betas=(0.5, 0.999),
        )

        loss_fn = nn.BCELoss()
        return generator, discriminator, g_optim, d_optim, loss_fn

    def train_discriminator_regression(
        self, optimizer, real_data, fake_data, discriminator, loss_fn
    ):
        # n = real_data.size(0)
        optimizer.zero_grad()
        prediction_real = discriminator(real_data)

        prediction_fake = discriminator(fake_data)

        D_loss = -(torch.mean(prediction_real) - torch.mean(prediction_fake))

        D_loss.backward()
        optimizer.step()

        for p in discriminator.parameters():
            p.data.clamp_(-0.005, 0.005)

        return D_loss

    def train_generator_regression(self, optimizer, fake_data, discriminator, loss_fn):
        optimizer.zero_grad()
        prediction = discriminator(fake_data)
        G_loss = -torch.mean(prediction)

        G_loss.backward()
        optimizer.step()

        return G_loss, optimizer

    def create_gan_train_dataset_regression(self):
        logging.info("Create NLP train dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        X_train[self.target_variable] = Y_train
        self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

        train_dataset = GanDataset(
            torch.from_numpy(X_train.to_numpy().astype(float)).float(),
            torch.from_numpy(Y_train.to_numpy().astype(float)).float(),
        )
        return train_dataset

    def create_gan_train_dataloader_regression(
        self, train_batch_size=None, workers=None
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
        train_dataset = self.create_gan_train_dataset_regression()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        return train_dataloader

    def create_gan_test_dataloader_regression(
        self, train_batch_size=None, workers=None
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
        train_dataset = self.create_gan_test_dataset_regression()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        return train_dataloader

    def train_gan_regression(self):
        epochs = self.gan_settings["max_epochs"]
        k = self.gan_settings["discriminator_extra_training_rounds"]
        best_epoch = -1
        # test_noise = self.noise(64)
        train_dataloader = self.create_gan_train_dataloader_regression()
        (
            generator,
            discriminator,
            g_optim,
            d_optim,
            loss_fn,
        ) = self.gan_model_setup_regression()

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
                    d_loss += self.train_discriminator_regression(
                        d_optim,
                        real_data,
                        fake_data,
                        discriminator=discriminator,
                        loss_fn=loss_fn,
                    )
                fake_data = generator(self.noise(n))
                g_loss_it, g_optim = self.train_generator_regression(
                    g_optim, fake_data, discriminator=discriminator, loss_fn=loss_fn
                )
                g_loss = g_loss + g_loss_it

                del fake_data
                del real_data
                _ = gc.collect()

                # img = generator(test_noise).cpu().detach()
                g_losses.append(g_loss / i)
                d_losses.append(d_loss / i)

            if epoch == 0 or g_loss < best_loss:
                print(
                    f"Found better model in epoch {epoch} with generator loss {g_loss}"
                    f" and discriminator loss {d_loss}."
                )
                best_epoch = epoch
                best_loss = g_loss
                state = {
                    "state_dict": generator.state_dict(),
                    "optimizer_dict": g_optim.state_dict(),
                    "bestscore": g_loss,
                }
                torch.save(state, "generator_model.pth")

            if epoch > best_epoch + self.gan_settings["early_stopping_rounds"]:
                break

                # clear_output()
        del discriminator
        del generator
        torch.cuda.empty_cache()
        _ = gc.collect()

    def train_regression_generators(self):
        self.get_current_timestamp(task="Start training GANs.")
        self.train_gan_regression()
        self.get_current_timestamp(task="Finished training GANs.")

    def load_generator_model_states_regression(self, path=None):
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
                if "generator_model.pth" in s
            ]
            return pthes
        else:
            pthes = [
                os.path.join(f"{path}/", s)
                for s in os.listdir(f"{path}/")
                if "generator_model.pth" in s
            ]
            return pthes

    def create_synthetic_data_regression(self):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        columns = X_train.columns

        rows_to_create = self.gan_settings["nb_synthetic_rows_to_create"]
        batch_size = self.gan_settings["batch_size"]
        nb_batches = int(rows_to_create / batch_size)
        new_data = []
        (
            generator,
            discriminator,
            g_optim,
            d_optim,
            loss_fn,
        ) = self.gan_model_setup_regression()
        pathes = self.load_generator_model_states_regression(
            path=self.gan_model_save_states_path
        )
        print(pathes)
        for m_path in pathes:
            print(f"Load {m_path}")
            state = torch.load(m_path)
            generator.load_state_dict(state["state_dict"])
            generator.to(device)
            generator.eval()

            with torch.no_grad():
                for _batch in range(nb_batches):
                    synth_data = generator.predict(self.noise(batch_size))
                    synth_data = synth_data.cpu().detach()
                    new_data.append(synth_data)

            del state
            torch.cuda.empty_cache()
            _ = gc.collect()

        if self.gan_settings["concat_to_original_data"]:
            X_train[self.target_variable] = Y_train
            X_train = X_train.sort_index(axis=1)
            original_columns = X_train.columns
            full_new_data = np.concatenate(new_data, axis=0)
            X_train_synth = pd.DataFrame(full_new_data, columns=columns)
            X_train_synth = X_train_synth.sort_index(axis=1)
            X_train = pd.concat(
                [X_train[original_columns], X_train_synth[original_columns]]
            )
        else:
            full_new_data = np.concatenate(new_data, axis=0)
            X_train = pd.DataFrame(full_new_data, columns=columns)

        X_train = X_train.reset_index(drop=True)

        Y_train = X_train[self.target_variable]
        X_train = X_train.drop(self.target_variable, axis=1)
        print(f"Synthetic train df is of length {len(X_train.index)}")

        try:
            X_test = X_test.drop(self.target_variable, axis=1)
        except KeyError:
            pass

        del generator

        self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
