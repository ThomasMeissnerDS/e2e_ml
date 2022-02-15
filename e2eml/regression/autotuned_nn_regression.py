import gc
import logging
import os

import numpy as np
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, median_absolute_error
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from e2eml.full_processing import cpu_preprocessing, postprocessing

# specify GPU
scaler = torch.cuda.amp.GradScaler()  # GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegressionDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class RegressionNNModel(
    postprocessing.FullPipeline, cpu_preprocessing.PreProcessing, RegressionDataset
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

    def create_nn_train_dataset(self):
        logging.info("Create NLP train dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        train_dataset = RegressionDataset(
            torch.from_numpy(X_train.to_numpy()).float(),
            torch.from_numpy(Y_train.to_numpy()).float(),
        )
        return train_dataset

    def create_nn_test_dataset(self):
        logging.info("Create NLP test dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        test_dataset = RegressionDataset(
            torch.from_numpy(X_test.to_numpy()).float(),
            torch.from_numpy(Y_test.to_numpy()).float(),
        )
        return test_dataset

    def create_nn_pred_dataset(self):
        logging.info("Create NLP prediction dataset.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        if self.prediction_mode:
            self.dataframe[self.target_variable] = 999  # creating dummy column
            dummy_target = self.dataframe[self.target_variable]
            self.dataframe = self.dataframe.drop(self.target_variable, axis=1)
            pred_dataset = RegressionDataset(
                torch.from_numpy(self.dataframe.to_numpy()).float(),
                torch.from_numpy(dummy_target.to_numpy()).float(),
            )
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            pred_dataset = RegressionDataset(
                torch.from_numpy(X_test.to_numpy()).float(),
                torch.from_numpy(Y_test.to_numpy()).float(),
            )
        return pred_dataset

    def create_nn_train_dataloader(self, train_batch_size=None, workers=None):
        if train_batch_size:
            pass
        else:
            train_batch_size = self.autotuned_nn_settings["train_batch_size"]

        if workers:
            pass
        else:
            workers = self.autotuned_nn_settings["num_workers"]
        logging.info("Create Neural network train dataloader.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        train_dataset = self.create_nn_train_dataset()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        return train_dataloader

    def create_nn_test_dataloader(self, test_batch_size=None, workers=None):
        if test_batch_size:
            pass
        else:
            test_batch_size = self.autotuned_nn_settings["test_batch_size"]

        if workers:
            pass
        else:
            workers = self.autotuned_nn_settings["num_workers"]
        logging.info("Create NLP test dataloader.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        test_dataset = self.create_nn_test_dataset()
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        return test_dataloader

    def pred_nn_dataloader(self, pred_batch_size=None, workers=None):
        if pred_batch_size:
            pass
        else:
            pred_batch_size = self.autotuned_nn_settings["pred_batch_size"]

        if workers:
            pass
        else:
            workers = self.autotuned_nn_settings["num_workers"]
        logging.info("Create Neural network prediction dataloader.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        pred_dataset = self.create_nn_pred_dataset()
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=pred_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )
        return pred_dataloader

    def loss_fn(self, output, target):
        return torch.sqrt(nn.MSELoss()(output, target))

    def get_num_features(self):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        N, D = X_train.shape
        self.preprocess_decisions["num_features"] = D
        return D

    def get_nn_architecture(self, type="ann", num_features=1):
        if type == "ann":

            class MultipleRegression(nn.Module):
                def __init__(self, num_features):
                    super(MultipleRegression, self).__init__()

                    self.layer_1 = nn.Linear(num_features, 64)
                    self.layer_2 = nn.Linear(64, 128)
                    self.layer_3 = nn.Linear(128, 64)
                    self.layer_out = nn.Linear(64, 1)

                    self.relu = nn.ReLU()

                def forward(self, inputs):
                    x = self.relu(self.layer_1(inputs))
                    x = self.relu(self.layer_2(x))
                    x = self.relu(self.layer_3(x))
                    x = self.layer_out(x)
                    return x

                def predict(self, test_inputs):
                    x = self.relu(self.layer_1(test_inputs))
                    x = self.relu(self.layer_2(x))
                    x = self.relu(self.layer_3(x))
                    x = self.layer_out(x)
                    return x

            model = MultipleRegression(num_features=num_features)
            return model
        elif type == "1d-cnn":

            class SoftOrdering1DCNN(nn.Module):
                def __init__(
                    self,
                    input_dim,
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

                    self.batch_norm1 = nn.BatchNorm1d(input_dim)
                    self.dropout1 = nn.Dropout(dropout_input)
                    dense1 = nn.Linear(input_dim, hidden_size, bias=False)
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

            model = SoftOrdering1DCNN(input_dim=num_features)
            return model

        else:

            class MultipleRegression(nn.Module):
                def __init__(self, num_features):
                    super(MultipleRegression, self).__init__()

                    self.layer_1 = nn.Linear(num_features, 64)
                    self.layer_2 = nn.Linear(64, 128)
                    self.layer_3 = nn.Linear(128, 64)
                    self.layer_out = nn.Linear(64, 1)

                    self.relu = nn.ReLU()

                def forward(self, inputs):
                    x = self.relu(self.layer_1(inputs))
                    x = self.relu(self.layer_2(x))
                    x = self.relu(self.layer_3(x))
                    x = self.layer_out(x)
                    return x

                def predict(self, test_inputs):
                    x = self.relu(self.layer_1(test_inputs))
                    x = self.relu(self.layer_2(x))
                    x = self.relu(self.layer_3(x))
                    x = self.layer_out(x)
                    return x

            model = MultipleRegression(num_features=num_features)
            return model

    # nn.MultiMarginLoss, #CrossEntropyLoss, #MSELoss

    def nn_model_setup(self, epochs=None):
        if self.prediction_mode:
            pass
        else:
            logging.info("Define NLP model.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.get_num_features()
            model = self.get_nn_architecture(
                type=self.autotuned_nn_settings["architecture"],
                num_features=self.preprocess_decisions["num_features"],
            )
            model.to(device)
            model.train()
            LR = 2e-5  # 1e-3
            optimizer = AdamW(
                model.parameters(), LR, betas=(0.99, 0.998), weight_decay=1e-2
            )
            if epochs:
                pass
            else:
                epochs = self.autotuned_nn_settings["epochs"]
            epochs = epochs
            train_steps = int(
                len(X_train) / self.autotuned_nn_settings["train_batch_size"] * epochs
            )
            num_steps = int(train_steps * 0.1)
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_steps, train_steps
            )
            return model, optimizer, train_steps, num_steps, scheduler

    def nn_training(self, train_dataloader, model, optimizer, scheduler):
        logging.info("Start NLP training loop.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        model.train()
        allpreds = []
        alltargets = []
        self.reset_test_train_index(drop_target=True)

        for X_train_batch, y_train_batch in train_dataloader:
            losses = []
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                X_train_batch, y_train_batch = X_train_batch.to(
                    device
                ), y_train_batch.to(device)

                y_train_pred = model(X_train_batch)
                y_train_batch = y_train_batch.unsqueeze(1)
                loss = self.loss_fn(y_train_pred, y_train_batch)

                # For scoring
                losses.append(loss.item() / len(y_train_pred))
                allpreds.append(y_train_pred.detach().cpu().numpy())
                alltargets.append(y_train_batch.detach().squeeze(-1).cpu().numpy())

            scaler.scale(loss).backward()  # backwards of loss
            scaler.step(optimizer)  # Update optimizer
            scaler.update()  # scaler update
            scheduler.step()  # Update learning rate schedule

            # Combine dataloader minutes

        allpreds = np.concatenate(allpreds)
        alltargets = np.concatenate(alltargets)

        # I don't use loss, but I collect it
        losses = np.mean(losses)
        # Score with rmse
        train_rme_loss = np.sqrt(mean_squared_error(alltargets, allpreds))

        return losses, train_rme_loss

    def nn_validating(self, valid_dataloader, model):
        logging.info("Start NLP validation loop.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        model.eval()
        allpreds = []
        alltargets = []

        for X_test_batch, y_test_batch in valid_dataloader:
            losses = []

            with torch.no_grad():
                X_test_batch, y_test_batch = X_test_batch.to(device), y_test_batch.to(
                    device
                )

                y_train_pred = model(X_test_batch)
                y_test_batch = y_test_batch.unsqueeze(1)
                loss = self.loss_fn(y_train_pred, y_test_batch)

                # For scoring
                losses.append(loss.item() / len(y_train_pred))
                allpreds.append(y_train_pred.detach().cpu().numpy())
                alltargets.append(y_test_batch.detach().squeeze(-1).cpu().numpy())
                # Combine dataloader minutes

        allpreds = np.concatenate(allpreds)
        alltargets = np.concatenate(alltargets)

        # I don't use loss, but I collect it
        losses = np.mean(losses)
        # Score with rmse
        valid_rme_loss = np.sqrt(mean_squared_error(alltargets, allpreds))

        return allpreds, losses, valid_rme_loss

    def nn_predicting(self, pred_dataloader, model, pathes):
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
            # allvalloss = 0
            with torch.no_grad():
                for X_pred_batch, y_pred_batch in pred_dataloader:
                    X_pred_batch, y_pred_batch = X_pred_batch.to(
                        device
                    ), y_pred_batch.to(device)
                    y_pred_pred = model(X_pred_batch)
                    y_pred_pred = y_pred_pred.squeeze(-1)
                    preds.append(y_pred_pred.detach().cpu().numpy())

                preds = np.concatenate(preds)

                if self.prediction_mode:
                    self.dataframe[f"preds_model{model_no}"] = preds
                else:
                    X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
                    X_test[f"preds_model{model_no}"] = preds
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

    def neural_network_train(self):
        if self.prediction_mode:
            pass
        else:
            logging.info("Start NLP transformer training.")
            logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
            self.reset_test_train_index(drop_target=True)

            train_dataloader = self.create_nn_train_dataloader()
            test_dataloader = self.create_nn_test_dataloader()
            model, optimizer, train_steps, num_steps, scheduler = self.nn_model_setup()
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_steps, train_steps
            )

            trainlosses = []
            vallosses = []
            bestscore = None
            trainscores = []
            validscores = []

            for epoch in tqdm(range(self.autotuned_nn_settings["epochs"])):
                print("---------------" + str(epoch) + "start-------------")
                trainloss, trainscore = self.nn_training(
                    train_dataloader, model, optimizer, scheduler
                )
                trainlosses.append(trainloss)
                trainscores.append(trainscore)
                print("trainscore is " + str(trainscore))
                preds, validloss, valscore = self.nn_validating(test_dataloader, model)
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

            for fold in range(1, 5):

                self.reset_test_train_index(drop_target=True)
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

                # initializing the data
                train_dataloader = self.create_nn_train_dataloader()
                test_dataloader = self.create_nn_test_dataloader()

                model = self.get_nn_architecture(
                    type=self.autotuned_nn_settings["architecture"],
                    num_features=self.preprocess_decisions["num_features"],
                )
                model.to(device)
                LR = 2e-5  # 1e-3
                optimizer = AdamW(
                    model.parameters(), LR, betas=(0.99, 0.999), weight_decay=1e-2
                )
                train_steps = int(
                    len(X_train)
                    / self.autotuned_nn_settings["train_batch_size"]
                    * self.autotuned_nn_settings["epochs"]
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

                for epoch in tqdm(range(self.autotuned_nn_settings["epochs"])):
                    print("---------------" + str(epoch) + "start-------------")
                    trainloss, trainscore = self.nn_training(
                        train_dataloader, model, optimizer, scheduler
                    )
                    trainlosses.append(trainloss)
                    trainscores.append(trainscore)

                    print("trainscore is " + str(trainscore))
                    preds, validloss, valscore = self.nn_validating(
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

    def median_abs_error_eval(self, true_y, predicted):
        try:
            median_absolute_error_score = median_absolute_error(true_y, predicted)
        except Exception:
            median_absolute_error_score = 0
        return median_absolute_error_score

    def neural_network_predict(self):
        logging.info("Start NLP transformer prediction.")
        logging.info(f"RAM memory {psutil.virtual_memory()[2]} percent used.")
        self.reset_test_train_index(drop_target=True)
        model = self.get_nn_architecture(
            type=self.autotuned_nn_settings["architecture"],
            num_features=self.preprocess_decisions["num_features"],
        )
        pthes = self.load_model_states(path=self.transformer_model_save_states_path)
        print(pthes)
        pred_dataloader = self.pred_nn_dataloader()
        allpreds, mode_cols = self.nn_predicting(pred_dataloader, model, pthes)

        if self.prediction_mode:
            self.dataframe["neural_network_median"] = self.dataframe[mode_cols].median(
                axis=1
            )
            self.dataframe["neural_network_mean"] = self.dataframe[mode_cols].mean(
                axis=1
            )
            self.predicted_values["neural_network"] = self.dataframe[
                "neural_network_mean"
            ]
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            if self.autotuned_nn_settings["keep_best_model_only"]:
                # we check, if one savestate underperforms and delete him out
                scorings = []
                states = []
                for state in mode_cols:
                    state_score = self.median_abs_error_eval(Y_test, X_test[state])
                    # print(state_score)
                    scorings.append(state_score)
                    states.append(state)
                scorings_arr = np.array(scorings)
                scorings_mean = np.mean(scorings_arr)
                scorings_std = np.std(scorings_arr)
                # print(scorings_std)
                keep_state = scorings_arr < scorings_mean + scorings_std
                for index, state in enumerate(states):
                    os_string = state[-6:]
                    if keep_state.tolist()[index]:
                        pass
                    else:
                        states.remove(state)
                        X_test.drop(state, axis=1)
                        os.remove(f"{os_string}.pth")
            else:
                pass

            X_test["neural_network_median"] = X_test[mode_cols].median(axis=1)
            X_test["neural_network_mean"] = X_test[mode_cols].mean(axis=1)
            self.predicted_values["neural_network"] = X_test["neural_network_median"]
