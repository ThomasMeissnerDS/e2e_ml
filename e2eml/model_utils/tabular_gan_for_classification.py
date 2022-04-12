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

            def __init__(
                self,
                num_features,
                output_dim=1,
                sign_size=32,
                cha_input=16,
                cha_hidden=32,
                K=2,
                dropout_input=0.2,
                dropout_hidden=0.2,
                dropout_output=0.2,
            ):
                super().__init__()

                hidden_size = sign_size * cha_input
                sign_size1 = sign_size
                sign_size2 = sign_size // 2
                output_size = (sign_size // 4) * cha_hidden

                self.hidden_size = hidden_size
                self.cha_input = cha_input
                self.cha_hidden = cha_hidden
                self.K = K
                self.sign_size1 = sign_size1
                self.sign_size2 = sign_size2
                self.output_size = output_size
                self.dropout_input = dropout_input
                self.dropout_hidden = dropout_hidden
                self.dropout_output = dropout_output

                self.batch_norm1 = nn.BatchNorm1d(num_features)
                self.dropout1 = nn.Dropout(dropout_input)
                dense1 = nn.Linear(num_features, hidden_size, bias=False)
                self.dense1 = nn.utils.weight_norm(dense1)

                # 1st conv layer
                self.batch_norm_c1 = nn.BatchNorm1d(cha_input)
                conv1 = nn.Conv1d(
                    cha_input,
                    cha_input * K,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    groups=cha_input,
                    bias=False,
                )
                self.conv1 = nn.utils.weight_norm(conv1, dim=None)

                self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=sign_size2)

                # 2nd conv layer
                self.batch_norm_c2 = nn.BatchNorm1d(cha_input * K)
                self.dropout_c2 = nn.Dropout(dropout_hidden)
                conv2 = nn.Conv1d(
                    cha_input * K,
                    cha_hidden,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                self.conv2 = nn.utils.weight_norm(conv2, dim=None)

                # 3rd conv layer
                self.batch_norm_c3 = nn.BatchNorm1d(cha_hidden)
                self.dropout_c3 = nn.Dropout(dropout_hidden)
                conv3 = nn.Conv1d(
                    cha_hidden,
                    cha_hidden,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                self.conv3 = nn.utils.weight_norm(conv3, dim=None)

                # 4th conv layer
                self.batch_norm_c4 = nn.BatchNorm1d(cha_hidden)
                conv4 = nn.Conv1d(
                    cha_hidden,
                    cha_hidden,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    groups=cha_hidden,
                    bias=False,
                )
                self.conv4 = nn.utils.weight_norm(conv4, dim=None)

                self.avg_po_c4 = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)

                self.flt = nn.Flatten()

                self.batch_norm2 = nn.BatchNorm1d(output_size)
                self.dropout2 = nn.Dropout(dropout_output)
                dense2 = nn.Linear(output_size, output_dim, bias=False)
                self.dense2 = nn.utils.weight_norm(dense2)

                # self.loss = nn.BCEWithLogitsLoss()

            def forward(self, x):
                x = self.batch_norm1(x)
                x = self.dropout1(x)
                x = nn.functional.celu(self.dense1(x))

                x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

                x = self.batch_norm_c1(x)
                x = nn.functional.relu(self.conv1(x))

                x = self.ave_po_c1(x)

                x = self.batch_norm_c2(x)
                x = self.dropout_c2(x)
                x = nn.functional.relu(self.conv2(x))
                x_s = x

                x = self.batch_norm_c3(x)
                x = self.dropout_c3(x)
                x = nn.functional.relu(self.conv3(x))

                x = self.batch_norm_c4(x)
                x = self.conv4(x)
                x = x + x_s
                x = nn.functional.relu(x)

                x = self.avg_po_c4(x)

                x = self.flt(x)

                x = self.batch_norm2(x)
                x = self.dropout2(x)
                x = self.dense2(x)

                return x

            def predict(self, x):
                x = self.batch_norm1(x)
                x = self.dropout1(x)
                x = nn.functional.celu(self.dense1(x))

                x = x.reshape(x.shape[0], self.cha_input, self.sign_size1)

                x = self.batch_norm_c1(x)
                x = nn.functional.relu(self.conv1(x))

                x = self.ave_po_c1(x)

                x = self.batch_norm_c2(x)
                x = self.dropout_c2(x)
                x = nn.functional.relu(self.conv2(x))
                x_s = x

                x = self.batch_norm_c3(x)
                x = self.dropout_c3(x)
                x = nn.functional.relu(self.conv3(x))

                x = self.batch_norm_c4(x)
                x = self.conv4(x)
                x = x + x_s
                x = nn.functional.relu(x)

                x = self.avg_po_c4(x)

                x = self.flt(x)

                x = self.batch_norm2(x)
                x = self.dropout2(x)
                x = self.dense2(x)

                return x

        class Discriminator(nn.Module):
            """
            Will decide, if data is real or fake.
            """

            def __init__(self, num_features, dropout=0.3):
                super(Discriminator, self).__init__()

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
        n_features = X_train.values.shape[1]
        generator = Generator(
            num_features=n_features,
            output_dim=n_features,
            dropout_input=self.gan_settings["generator_dropout_rate"],
            dropout_hidden=self.gan_settings["generator_dropout_rate"],
            dropout_output=self.gan_settings["generator_dropout_rate"],
        )
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

        loss_fn = nn.BCEWithLogitsLoss()
        return generator, discriminator, g_optim, d_optim, loss_fn

    def boundary_seeking_loss(self, y_pred, y_true):
        return 0.5 * torch.mean((torch.log(y_pred) - torch.log(1 - y_pred)) ** 2)

    def train_discriminator(
        self, optimizer, real_data, fake_data, discriminator, loss_fn
    ):
        n = real_data.size(0)
        optimizer.zero_grad()
        real_pred = discriminator(real_data)
        fake_pred = discriminator(fake_data.detach())
        real_loss = loss_fn(real_pred - fake_pred, self.make_ones(n))
        fake_loss = loss_fn(fake_pred - real_pred, self.make_zeros(n))
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer.step()

        return d_loss

    def train_generator(self, optimizer, fake_data, real_data, discriminator, loss_fn):
        n = fake_data.size(0)
        optimizer.zero_grad()

        discriminator(real_data).detach()
        discriminator(fake_data)
        g_loss = loss_fn(discriminator(fake_data), self.make_ones(n))

        g_loss.backward()
        optimizer.step()

        return g_loss, optimizer

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
                real_data = imgs.to(device)
                g_loss_it, g_optim = self.train_generator(
                    g_optim,
                    fake_data,
                    real_data,
                    discriminator=discriminator,
                    loss_fn=loss_fn,
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
                torch.save(state, f"class_{target_class}_generator_model.pth")
                # clear_output()

            if epoch > best_epoch + self.gan_settings["early_stopping_rounds"]:
                break

        del discriminator
        del generator
        torch.cuda.empty_cache()
        _ = gc.collect()

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

        if self.gan_settings["concat_to_original_data"]:
            X_train[self.target_variable] = Y_train
            X_train = X_train.sort_index(axis=1)
            print(X_train.head(3))
            original_columns = X_train.columns
            X_train_synth = pd.concat(new_class_data)
            Y_train_synth = pd.Series(np.concatenate(new_targets, axis=0))
            print(X_train_synth.head(3))
            X_train_synth[self.target_variable] = Y_train_synth.values
            X_train = pd.concat(
                [X_train[original_columns], X_train_synth[original_columns]]
            )
        else:
            X_train = pd.concat(new_class_data)
            Y_train = pd.Series(np.concatenate(new_targets, axis=0))
            X_train[self.target_variable] = Y_train.values

        if len(X_train.index) > rows_to_create:
            X_train = X_train.sample(
                rows_to_create, random_state=self.global_random_state
            )

        X_train = X_train.reset_index(drop=True)
        Y_train = X_train[self.target_variable].copy()
        X_train = X_train.drop(self.target_variable, axis=1)

        try:
            X_test = X_test.drop(self.target_variable, axis=1)
        except KeyError:
            pass

        self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
