import gc
import logging
import os
import re

import numpy as np
import psutil
import torch
import torch.nn as nn
from numpy import inf
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from e2eml.full_processing import cpu_preprocessing, postprocessing

# specify GPU
scaler = torch.cuda.amp.GradScaler()  # GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.from_numpy(dataframe[target].to_numpy(dtype=np.float32)).float()
        self.X = torch.from_numpy(
            dataframe[features].to_numpy(dtype=np.float32)
        ).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1).float()
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0).float()

        return x, self.y[i]


class LSTMQuantileModel(
    postprocessing.FullPipeline, cpu_preprocessing.PreProcessing, TimeSeriesDataset
):
    """
    This class stores all model training and prediction methods for classification tasks.
    This class stores all pipeline relevant information (inherited from cpu preprocessing).
    The attribute "df_dict" always holds train and test as well as
    to predict data. The attribute "preprocess_decisions" stores encoders and other information generated during the
    model training. The attributes "predicted_classes" and "predicted_probs" store dictionaries (model names are dictionary keys)
    with predicted classes and probabilities (classification tasks) while "predicted_values" stores regression based
    predictions. The attribute "evaluation_scores" keeps track of model evaluation metrics (in dictionary format).
    :param datasource: Expects a Pandas dataframe (containing the target feature as a column)
    :param target_variable: Name of the target feature's column within the datasource dataframe.
    :param date_columns: Date columns can be passed as lists additionally for respective preprocessing. If not provided
    e2eml will try to detect datetime columns automatically. Date format is expected as YYYY-MM-DD anyway.
    :param categorical_columns: Categorical columns can be passed as lists additionally for respective preprocessing.
    If not provided e2eml will try to detect categorical columns automatically.
    :param nlp_columns: NLP columns expect a string declaring one text column.
    :param unique_identifier: A unique identifier (i.e. an ID column) can be passed as well to preserve this information
     for later processing.
    :param ml_task: Can be 'binary', 'multiclass' or 'regression'. On default will be determined automatically.
    :param preferred_training_mode: Must be 'cpu', if e2eml has been installed into an environment without LGBM and Xgboost on GPU.
    Can be set to 'gpu', if LGBM and Xgboost have been installed with GPU support. The default 'auto' will detect GPU support
    and optimize accordingly. (Default: 'auto')
    :param logging_file_path: Preferred location to save the log file. Will otherwise stored in the current folder.
    :param low_memory_mode: Adds a preprocessing feature to reduce dataframe memory footprint. Will lead to a loss in
    model performance. Will be extended by further memory savings features in future releases.
    However we highly recommend GPU usage to heavily decrease model training times.
    """

    def create_lstm_quantile_train_dataset(self):
        logging.info("Create NLP train dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        train_dataset = TimeSeriesDataset(
            X_train[self.preprocess_decisions["n_features"]],
            self.target_variable,
            list(
                self.preprocess_decisions["n_features"].difference(
                    [self.target_variable]
                )
            ),
            sequence_length=self.lstm_settings["seq_len"],
        )
        return train_dataset

    def create_lstm_quantile_test_dataset(self):
        logging.info("Create NLP test dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        test_dataset = TimeSeriesDataset(
            X_test[self.preprocess_decisions["n_features"]],
            self.target_variable,
            list(
                self.preprocess_decisions["n_features"].difference(
                    [self.target_variable]
                )
            ),
            sequence_length=self.lstm_settings["seq_len"],
        )
        return test_dataset

    def create_lstm_quantile_pred_dataset(self):
        logging.info("Create NLP prediction dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            self.dataframe[self.target_variable] = 999  # creating dummy column
            # dummy_target = self.dataframe[self.target_variable]
            # self.dataframe = self.dataframe.drop(self.target_variable, axis=1)
            pred_dataset = TimeSeriesDataset(
                self.dataframe[self.preprocess_decisions["n_features"]],
                self.target_variable,
                list(
                    self.preprocess_decisions["n_features"].difference(
                        [self.target_variable]
                    )
                ),
                sequence_length=self.lstm_settings["seq_len"],
            )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            pred_dataset = TimeSeriesDataset(
                X_test[self.preprocess_decisions["n_features"]],
                self.target_variable,
                list(
                    self.preprocess_decisions["n_features"].difference(
                        [self.target_variable]
                    )
                ),
                sequence_length=self.lstm_settings["seq_len"],
            )
        return pred_dataset

    def create_lstm_quantile_train_dataloader(
        self, train_batch_size=None, workers=None
    ):
        if train_batch_size:
            pass
        else:
            train_batch_size = self.lstm_settings["train_batch_size"]

        if workers:
            pass
        else:
            workers = self.lstm_settings["num_workers"]
        logging.info("Create Neural network train dataloader.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        train_dataset = self.create_lstm_quantile_train_dataset()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        return train_dataloader

    def create_lstm_quantile_test_dataloader(self, test_batch_size=None, workers=None):
        if test_batch_size:
            pass
        else:
            test_batch_size = self.lstm_settings["test_batch_size"]

        if workers:
            pass
        else:
            workers = self.lstm_settings["num_workers"]
        logging.info("Create NLP test dataloader.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        test_dataset = self.create_lstm_quantile_test_dataset()
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        return test_dataloader

    def pred_lstm_quantile_dataloader(self, pred_batch_size=None, workers=None):
        if pred_batch_size:
            pass
        else:
            pred_batch_size = self.lstm_settings["pred_batch_size"]

        if workers:
            pass
        else:
            workers = self.lstm_settings["num_workers"]
        logging.info("Create Neural network prediction dataloader.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        pred_dataset = self.create_lstm_quantile_pred_dataset()
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=pred_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        return pred_dataloader

    def get_num_features(self):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        N, D = X_train.shape
        self.preprocess_decisions["num_features"] = D
        return D

    def get_lstm_quantile_architecture(
        self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob
    ):
        class lstmQuantile(nn.Module):
            def __init__(
                self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob
            ):
                super(lstmQuantile, self).__init__()

                # Defining the number of layers and the nodes in each layer
                self.hidden_dim = hidden_dim
                self.layer_dim = layer_dim

                self.lstm = nn.LSTM(
                    input_dim,
                    hidden_dim,
                    num_layers=layer_dim,
                    batch_first=True,
                    dropout=dropout_prob,
                )

                # Fully connected layer
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                # Initializing hidden state for first input with zeros
                h0 = torch.zeros(
                    self.layer_dim, x.size(0), self.hidden_dim, dtype=torch.float32
                ).to(device)

                # Initializing cell state for first input with zeros
                c0 = torch.zeros(
                    self.layer_dim, x.size(0), self.hidden_dim, dtype=torch.float32
                ).to(device)

                # We need to detach as we are doing truncated backpropagation through time (BPTT)
                # If we don't, we'll backprop all the way to the start even after going through another batch
                # Forward propagation by passing in the input, hidden state, and cell state into the model
                # Propagate input through LSTM
                ula, (h_out, _) = self.lstm(x, (h0.detach(), c0.detach()))

                h_out = h_out[0].view(-1, self.hidden_dim)

                out = self.fc(h_out)

                return out

        model = lstmQuantile(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
        return model

    def lstm_quantile_model_setup(self, epochs=None):
        if self.prediction_mode:
            pass
        else:
            logging.info("Define NLP model.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.get_num_features()
            model = self.get_lstm_quantile_architecture(
                input_dim=X_train[self.preprocess_decisions["n_features"]].shape[1]
                - 1,  # removes target from count
                hidden_dim=self.lstm_settings["hidden_dim"],
                layer_dim=self.lstm_settings["layer_dim"],
                output_dim=3,
                dropout_prob=self.lstm_settings["drop_out"],
            )
            model.to(device)
            model.train()
            LR = self.lstm_settings["learning_rate"]
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LR,
                weight_decay=self.lstm_settings["weight_decay"],
            )

            if epochs:
                pass
            else:
                epochs = self.lstm_settings["epochs"]
            epochs = epochs
            train_steps = int(
                len(X_train) / self.lstm_settings["train_batch_size"] * epochs
            )
            num_steps = int(train_steps * 0.1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_steps, train_steps
            )
            return model, optimizer, train_steps, num_steps, scheduler

    def simple_multi_bound_loss(self, y_hat_low, y_hat_up, target):
        mse_loss_mean = torch.mean((target - ((y_hat_low + y_hat_up) / 2)) ** 2)
        return mse_loss_mean

    def quantile_loss(self, preds, target, quantiles):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

    def fillna_inf(self, array):
        array = np.array(array)
        array[np.isnan(array)] = 0
        array[array == -inf] = 0
        array[array == inf] = 0
        return array

    def lstm_quantile_training(self, train_dataloader, model, optimizer, scheduler):
        logging.info("Start NLP training loop.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        model.train()
        allpreds = []
        allpreds_low = []
        allpreds_up = []
        alltargets = []
        self.reset_test_train_index(drop_target=False)

        for X_train_batch, y_train_batch in train_dataloader:
            losses = []
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                X_train_batch, y_train_batch = X_train_batch.to(
                    device
                ), y_train_batch.to(device)

                y_train_pred = model(X_train_batch)

                y_train_batch = y_train_batch.unsqueeze(1)
                loss = self.quantile_loss(
                    y_train_pred, y_train_batch, self.lstm_settings["quantiles"]
                )

                # For scoring
                losses.append(loss.item() / len(y_train_pred))
                y_train_pred = y_train_pred[:, 1:2]
                allpreds_low.append(y_train_pred[:, 0:1].detach().cpu().numpy())
                allpreds_up.append(y_train_pred[:, 2:3].detach().cpu().numpy())
                allpreds.append(y_train_pred.detach().cpu().numpy())
                alltargets.append(y_train_batch.detach().squeeze(-1).cpu().numpy())

            scaler.scale(loss).backward()  # backwards of loss
            scaler.step(optimizer)  # Update optimizer
            scaler.update()  # scaler update
            scheduler.step()  # Update learning rate schedule

            # Combine dataloader minutes

        allpreds = np.concatenate(allpreds)
        alltargets = np.concatenate(alltargets)

        allpreds = self.fillna_inf(allpreds)

        # I don't use loss, but I collect it
        losses = np.mean(losses)
        # Score with rmse
        try:
            train_rme_loss = mean_absolute_error(alltargets, allpreds)
        except ValueError:
            train_rme_loss = mean_absolute_error(alltargets, allpreds)
        return losses, train_rme_loss

    def lstm_quantile_validating(self, valid_dataloader, model):
        logging.info("Start NLP validation loop.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        model.eval()
        allpreds = []
        allpreds_low = []
        allpreds_up = []
        alltargets = []

        for X_test_batch, y_test_batch in valid_dataloader:
            losses = []

            with torch.no_grad():
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(
                    device
                )
                y_train_pred = model(X_test_batch)
                y_test_batch = y_test_batch.unsqueeze(1)
                loss = self.quantile_loss(
                    y_train_pred, y_test_batch, self.lstm_settings["quantiles"]
                )

                # For scoring
                losses.append(loss.item() / len(y_train_pred))
                y_train_pred = y_train_pred[:, 1:2]
                allpreds_low.append(y_train_pred[:, 0:1].detach().cpu().numpy())
                allpreds_up.append(y_train_pred[:, 2:3].detach().cpu().numpy())
                allpreds.append(y_train_pred.detach().cpu().numpy())
                alltargets.append(y_test_batch.detach().squeeze(-1).cpu().numpy())

        allpreds_low = np.concatenate(allpreds_low)
        allpreds_up = np.concatenate(allpreds_up)
        allpreds = np.concatenate(allpreds)
        alltargets = np.concatenate(alltargets)

        allpreds_low = self.fillna_inf(allpreds_low)
        allpreds_up = self.fillna_inf(allpreds_up)
        allpreds = self.fillna_inf(allpreds)

        # I don't use loss, but I collect it
        losses = np.mean(losses)
        # Score with rmse
        try:
            valid_rme_loss = mean_absolute_error(alltargets, allpreds)
            # valid_rme_loss = np.nan_to_num(valid_rme_loss, nan=0, neginf=0)
        except ValueError:
            allpreds_low = self.fillna_inf(allpreds_low)
            allpreds_up = self.fillna_inf(allpreds_up)
            valid_rme_loss = mean_absolute_error(alltargets, allpreds)

        return allpreds, losses, valid_rme_loss

    def lstm_quantile_predicting(self, pred_dataloader, model, pathes):
        logging.info("Start Neural network prediction loop.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        allpreds = []
        model_no = 0
        mode_cols = []
        for m_path in pathes:
            state = torch.load(m_path)
            model.load_state_dict(state["state_dict"])
            model.to(device)
            model.eval()
            preds = []
            preds_low = []
            preds_up = []
            # allvalloss = 0
            with torch.no_grad():
                for X_pred_batch, y_pred_batch in pred_dataloader:
                    X_pred_batch, y_pred_batch = X_pred_batch.to(
                        device
                    ), y_pred_batch.to(device)
                    y_pred_pred = model(X_pred_batch)

                    preds_low.append(y_pred_pred[:, 0:1].detach().cpu().numpy())
                    preds_up.append(y_pred_pred[:, 2:3].detach().cpu().numpy())
                    preds.append(y_pred_pred[:, 1:2].detach().cpu().numpy())

                preds = np.concatenate(preds)
                preds_low = np.concatenate(preds_low)
                preds_up = np.concatenate(preds_up)

                if self.prediction_mode:
                    self.dataframe[f"preds_model{model_no}"] = preds
                    self.dataframe[f"preds_low_model{model_no}"] = preds_low
                    self.dataframe[f"preds_up_model{model_no}"] = preds_up

                    self.dataframe[
                        f"preds_low_model{model_no}"
                    ] = self.target_skewness_handling(
                        preds_to_reconvert=self.dataframe[
                            f"preds_low_model{model_no}"
                        ].values,
                        mode="revert",
                    )
                    self.dataframe[
                        f"preds_up_model{model_no}"
                    ] = self.target_skewness_handling(
                        preds_to_reconvert=self.dataframe[
                            f"preds_up_model{model_no}"
                        ].values,
                        mode="revert",
                    )
                else:
                    X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                    X_test[f"preds_model{model_no}"] = preds
                    X_test[f"preds_low_model{model_no}"] = preds_low
                    X_test[f"preds_up_model{model_no}"] = preds_up

                    X_test[
                        f"preds_low_model{model_no}"
                    ] = self.target_skewness_handling(
                        preds_to_reconvert=X_test[f"preds_low_model{model_no}"].values,
                        mode="revert",
                    )
                    X_test[f"preds_up_model{model_no}"] = self.target_skewness_handling(
                        preds_to_reconvert=X_test[f"preds_up_model{model_no}"].values,
                        mode="revert",
                    )
                mode_cols.append(f"preds_model{model_no}")

                allpreds.append(preds)
                model_no += 1
            del state
            torch.cuda.empty_cache()
            _ = gc.collect()
        return allpreds, mode_cols

    def load_model_states(self, path=None):
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
                if ".pth" in s
            ]
            return pthes
        else:
            pthes = [
                os.path.join(f"{path}/", s)
                for s in os.listdir(f"{path}/")
                if ".pth" in s
            ]
            return pthes

    def lstm_quantile_regression_train(self):
        if self.prediction_mode:
            pass
        else:
            logging.info("Start NLP transformer training.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.reset_test_train_index(drop_target=False)

            train_dataloader = self.create_lstm_quantile_train_dataloader()
            test_dataloader = self.create_lstm_quantile_test_dataloader()
            (
                model,
                optimizer,
                train_steps,
                num_steps,
                scheduler,
            ) = self.lstm_quantile_model_setup()
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_steps, train_steps
            )

            trainlosses = []
            vallosses = []
            bestscore = None
            trainscores = []
            validscores = []

            for epoch in tqdm(range(self.lstm_settings["epochs"])):
                print("---------------" + str(epoch) + "start-------------")
                trainloss, trainscore = self.lstm_quantile_training(
                    train_dataloader, model, optimizer, scheduler
                )
                trainlosses.append(trainloss)
                trainscores.append(trainscore)
                print("trainscore is " + str(trainscore))
                preds, validloss, valscore = self.lstm_quantile_validating(
                    test_dataloader, model
                )
                vallosses.append(validloss)
                validscores.append(valscore)

                print("valscore is " + str(valscore))
                if bestscore is None:
                    bestscore = valscore
                    print("Save first model")
                    state = {
                        "state_dict": model.state_dict(),
                        "optimizer_dict": optimizer.state_dict(),
                        "bestscore": bestscore,
                    }
                    torch.save(state, "model0.pth")

                elif bestscore > valscore:
                    bestscore = valscore
                    print("found better point")
                    state = {
                        "state_dict": model.state_dict(),
                        "optimizer_dict": optimizer.state_dict(),
                        "bestscore": bestscore,
                    }
                    torch.save(state, "model0.pth")
                else:
                    pass

            bestscores = []
            bestscores.append(bestscore)

            for fold in range(self.lstm_settings["nb_model_to_create"]):

                self.reset_test_train_index(drop_target=False)
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

                # initializing the data
                train_dataloader = self.create_lstm_quantile_train_dataloader()
                test_dataloader = self.create_lstm_quantile_test_dataloader()

                model = self.get_lstm_quantile_architecture(
                    input_dim=X_train[self.preprocess_decisions["n_features"]].shape[1]
                    - 1,  # removes target from count
                    hidden_dim=self.lstm_settings["hidden_dim"],
                    layer_dim=self.lstm_settings["layer_dim"],
                    output_dim=3,
                    dropout_prob=self.lstm_settings["drop_out"],
                )
                model.to(device)
                LR = 2e-5  # 1e-3
                optimizer = AdamW(
                    model.parameters(), LR, betas=(0.99, 0.999), weight_decay=1e-2
                )
                train_steps = int(
                    len(X_train)
                    / self.lstm_settings["train_batch_size"]
                    * self.lstm_settings["epochs"]
                )
                num_steps = int(train_steps * 0.1)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_steps, train_steps
                )

                trainlosses = []
                vallosses = []
                bestscore = None
                trainscores = []
                validscores = []

                for epoch in tqdm(range(self.lstm_settings["epochs"])):
                    print("---------------" + str(epoch) + "start-------------")
                    trainloss, trainscore = self.lstm_quantile_training(
                        train_dataloader, model, optimizer, scheduler
                    )
                    trainlosses.append(trainloss)
                    trainscores.append(trainscore)

                    print("trainscore is " + str(trainscore))
                    preds, validloss, valscore = self.lstm_quantile_validating(
                        test_dataloader, model
                    )

                    vallosses.append(validloss)
                    validscores.append(valscore)
                    print("valscore is " + str(valscore))

                    if bestscore is None:
                        bestscore = valscore
                        print("Save first model")
                        state = {
                            "state_dict": model.state_dict(),
                            "optimizer_dict": optimizer.state_dict(),
                            "bestscore": bestscore,
                        }
                        torch.save(state, "model" + str(fold) + ".pth")
                    elif bestscore > valscore:
                        bestscore = valscore
                        print("found better point")
                        state = {
                            "state_dict": model.state_dict(),
                            "optimizer_dict": optimizer.state_dict(),
                            "bestscore": bestscore,
                        }
                        torch.save(state, "model" + str(fold) + ".pth")
                    else:
                        pass
                bestscores.append(bestscore)

            del model, optimizer, scheduler
            _ = gc.collect()

    def lstm_quantile_regression_predict(self):
        logging.info("Start NLP transformer prediction.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        self.reset_test_train_index(drop_target=False)
        if self.prediction_mode:
            model = self.get_lstm_quantile_architecture(
                input_dim=self.dataframe[self.preprocess_decisions["n_features"]].shape[
                    1
                ]
                - 1,  # removes target from count
                hidden_dim=self.lstm_settings["hidden_dim"],
                layer_dim=self.lstm_settings["layer_dim"],
                output_dim=3,
                dropout_prob=self.lstm_settings["drop_out"],
            )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.get_lstm_quantile_architecture(
                input_dim=X_test[self.preprocess_decisions["n_features"]].shape[1]
                - 1,  # removes target from count
                hidden_dim=self.lstm_settings["hidden_dim"],
                layer_dim=self.lstm_settings["layer_dim"],
                output_dim=3,
                dropout_prob=self.lstm_settings["drop_out"],
            )
        pathes = self.load_model_states(path=self.tabular_nn_model_save_states_path)
        pthes = self.load_model_states(path=self.tabular_nn_model_save_states_path)
        try:
            for path in pathes:
                if re.search("generator_model.pth", path):
                    pthes.remove(path)
                if re.search("autoencoder_model.pth", path):
                    pthes.remove(path)
        except Exception:
            pass
        print(pthes)
        pred_dataloader = self.pred_lstm_quantile_dataloader()
        allpreds, mode_cols = self.lstm_quantile_predicting(
            pred_dataloader, model, pthes
        )

        if self.prediction_mode:
            self.dataframe["lstm_quantile_regression_median"] = self.dataframe[
                mode_cols
            ].median(axis=1)
            self.dataframe["lstm_quantile_regression_mean"] = self.dataframe[
                mode_cols
            ].mean(axis=1)

            self.dataframe[self.target_variable] = self.dataframe[
                "lstm_quantile_regression_mean"
            ]
            self.scale_with_target(mode="reverse", drop_target=False)
            self.dataframe[self.target_variable] = self.target_skewness_handling(
                preds_to_reconvert=self.dataframe[self.target_variable].values,
                mode="revert",
            )
            self.predicted_values["lstm_quantile_regression"] = self.dataframe[
                self.target_variable
            ]

        else:
            X_test["lstm_quantile_regression_median"] = X_test[mode_cols].median(axis=1)
            X_test["lstm_quantile_regression_mean"] = X_test[mode_cols].mean(axis=1)

            X_test[self.target_variable] = X_test["lstm_quantile_regression_mean"]
            self.scale_with_target(mode="reverse", drop_target=False)
            X_test[self.target_variable] = self.target_skewness_handling(
                preds_to_reconvert=X_test[self.target_variable].values,
                mode="revert",
            )
            self.predicted_values["lstm_quantile_regression"] = X_test[
                self.target_variable
            ]
