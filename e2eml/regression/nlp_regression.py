import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import transformers
from transformers import AutoModel, BertTokenizerFast, AdamW, BertModel, RobertaModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import matthews_corrcoef, mean_squared_error, median_absolute_error
from tqdm import tqdm
import logging
import psutil
import os
import gc
from sklearn.utils.class_weight import compute_class_weight
from e2eml.full_processing import postprocessing, cpu_processing_nlp
import random

# specify GPU
scaler = torch.cuda.amp.GradScaler()  # GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BERTDataSet(Dataset):
    def __init__(self, sentences, targets, tokenizer, max_length):
        self.sentences = sentences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_sen_length = 300

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        target = self.targets[idx]
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_sen_length,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.float)
        }


"""class BERTClass(torch.nn.Module):
    def __init__(self, transformer):
        super(BERTClass, self).__init__()
        self.bert = AutoModel.from_pretrained(transformer
                                              )
        config = AutoConfig.from_pretrained(transformer)
        config.update({"output_hidden_states": True,
                       "hidden_dropout_prob": 0.30,
                       #"layer_norm_eps": 1e-7
                       })

        self.roberta = AutoModel.from_pretrained(transformer, config=config)

        self.attention = nn.Sequential(
            nn.Linear(768, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )

        self.regressor = nn.Sequential(
            nn.Linear(768, 1)
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        roberta_output = self.roberta(input_ids=input_ids,
                                      token_type_ids=token_type_ids,
                                      attention_mask=attention_mask)

        # There are a total of 13 layers of hidden states.
        # 1 for the embedding layer, and 12 for the 12 Roberta layers.
        # We take the hidden states from the last Roberta layer.
        last_layer_hidden_states = roberta_output.hidden_states[-1]

        # The number of cells is MAX_LEN.
        # The size of the hidden state of each cell is 768 (for roberta-base).
        # In order to condense hidden states of all cells to a context vector,
        # we compute a weighted average of the hidden states of all cells.
        # We compute the weight of each cell, using the attention neural network.
        weights = self.attention(last_layer_hidden_states)

        # weights.shape is BATCH_SIZE x MAX_LEN x 1
        # last_layer_hidden_states.shape is BATCH_SIZE x MAX_LEN x 768
        # Now we compute context_vector as the weighted average.
        # context_vector.shape is BATCH_SIZE x 768
        context_vector = torch.sum(weights * last_layer_hidden_states, dim=1)

        # Now we reduce the context vector to the prediction score.
        return self.regressor(context_vector)"""


class NlpModel(postprocessing.FullPipeline, cpu_processing_nlp.NlpPreprocessing, BERTDataSet):
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
    def create_train_dataset(self):
        logging.info('Create NLP train dataset.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        tokenizer = self.preprocess_decisions[f"nlp_transformers"][f"transformer_tokenizer_{self.transformer_chosen}"]
        train_dataset = BERTDataSet(X_train[self.nlp_transformer_columns], Y_train, tokenizer, self.preprocess_decisions[f"nlp_transformers"][f"max_sentence_len"])
        return train_dataset

    def create_test_dataset(self):
        logging.info('Create NLP test dataset.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        tokenizer = self.preprocess_decisions[f"nlp_transformers"][f"transformer_tokenizer_{self.transformer_chosen}"]
        test_dataset = BERTDataSet(X_test[self.nlp_transformer_columns], Y_test, tokenizer, self.preprocess_decisions[f"nlp_transformers"][f"max_sentence_len"])
        return test_dataset

    def create_pred_dataset(self):
        logging.info('Create NLP prediction dataset.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if self.prediction_mode:
            self.dataframe[self.target_variable] = 999  # creating dummy column
            dummy_target = self.dataframe[self.target_variable]
            self.dataframe.drop(self.target_variable, axis=1)
            tokenizer = self.preprocess_decisions[f"nlp_transformers"][f"transformer_tokenizer_{self.transformer_chosen}"]
            pred_dataset = BERTDataSet(self.dataframe[self.nlp_transformer_columns], dummy_target, tokenizer, self.preprocess_decisions[f"nlp_transformers"][f"max_sentence_len"])
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            tokenizer = self.preprocess_decisions[f"nlp_transformers"][f"transformer_tokenizer_{self.transformer_chosen}"]
            pred_dataset = BERTDataSet(X_test[self.nlp_transformer_columns], Y_test, tokenizer, self.preprocess_decisions[f"nlp_transformers"][f"max_sentence_len"])
        return pred_dataset

    def create_train_dataloader(self, train_batch_size=None, workers=None):
        if train_batch_size:
            pass
        else:
            train_batch_size = self.transformer_settings["train_batch_size"]

        if workers:
            pass
        else:
            workers = self.transformer_settings["num_workers"]
        logging.info('Create NLP train dataloader.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        train_dataset = self.create_train_dataset()
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                                      pin_memory=True)
        return train_dataloader

    def create_test_dataloader(self, test_batch_size=None, workers=None):
        if test_batch_size:
            pass
        else:
            test_batch_size = self.transformer_settings["test_batch_size"]

        if workers:
            pass
        else:
            workers = self.transformer_settings["num_workers"]
        logging.info('Create NLP test dataloader.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        test_dataset = self.create_test_dataset()
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=workers,
                                     pin_memory=True)
        return test_dataloader

    def pred_dataloader(self, pred_batch_size=None, workers=None):
        if pred_batch_size:
            pass
        else:
            pred_batch_size = self.transformer_settings["pred_batch_size"]

        if workers:
            pass
        else:
            workers = self.transformer_settings["num_workers"]
        logging.info('Create NLP prediction dataloader.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        pred_dataset = self.create_pred_dataset()
        pred_dataloader = DataLoader(pred_dataset, batch_size=pred_batch_size, shuffle=False, num_workers=workers,
                                     pin_memory=True)
        return pred_dataloader

    def loss_fn(self, output, target):
        return torch.sqrt(nn.MSELoss()(output, target))

    # nn.MultiMarginLoss, #CrossEntropyLoss, #MSELoss

    def model_setup(self, epochs=None):
        if self.prediction_mode:
            pass
        else:
            logging.info('Define NLP model.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            model = self.create_bert_regression_model(chosen_model=self.transformer_chosen)
            model.to(device)
            model.train()
            LR = 2e-5 #1e-3
            optimizer = AdamW(model.parameters(), LR, betas=(0.99, 0.998), weight_decay=1e-2)
            if epochs:
                pass
            else:
                epochs = self.transformer_settings["epochs"]
            epochs = epochs
            train_steps = int(len(X_train) / self.transformer_settings["train_batch_size"] * epochs)
            num_steps = int(train_steps * 0.1)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_steps, train_steps)
            self.preprocess_decisions[f"nlp_transformers"][f"sheduler_{self.transformer_chosen}"] = scheduler
            return model, optimizer, train_steps, num_steps, scheduler

    def training(self, train_dataloader, model, optimizer, scheduler):
        logging.info('Start NLP training loop.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        model.train()
        allpreds = []
        alltargets = []
        self.reset_test_train_index()

        for a in train_dataloader:
            losses = []
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                ids = a["ids"].to(device)
                mask = a["mask"].to(device)
                target = a["target"].to(device)
                token_type_ids = a["token_type_ids"].to(device)

                output = model(ids, mask)
                output = output[0].squeeze(-1)
                loss = self.loss_fn(output, target)

                # For scoring
                losses.append(loss.item() / len(output))
                allpreds.append(output.detach().cpu().numpy())
                alltargets.append(target.detach().squeeze(-1).cpu().numpy())

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

    def validating(self, valid_dataloader, model):
        logging.info('Start NLP validation loop.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        model.eval()
        allpreds = []
        alltargets = []

        for a in valid_dataloader:
            losses = []

            with torch.no_grad():
                ids = a["ids"].to(device)
                mask = a["mask"].to(device)
                target = a["target"].to(device)
                token_type_ids = a["token_type_ids"].to(device)

                output = model(ids, mask)
                output = output[0].squeeze(-1)
                loss = self.loss_fn(output, target)
                # For scoring
                losses.append(loss.item() / len(output))
                allpreds.append(output.detach().cpu().numpy())
                alltargets.append(target.detach().squeeze(-1).cpu().numpy())
                # Combine dataloader minutes

        allpreds = np.concatenate(allpreds)
        alltargets = np.concatenate(alltargets)

        # I don't use loss, but I collect it
        losses = np.mean(losses)
        # Score with rmse
        valid_rme_loss = np.sqrt(mean_squared_error(alltargets, allpreds))

        return allpreds, losses, valid_rme_loss

    def predicting(self, pred_dataloader, model, pathes):
        logging.info('Start NLP prediction loop.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        allpreds = []
        model_no = 0
        mode_cols = []
        for m_path in pathes:
            state = torch.load(m_path)
            model.load_state_dict(state["state_dict"])
            model.to(device)
            model.eval()
            preds = []
            allvalloss = 0
            with torch.no_grad():
                for a in pred_dataloader:
                    ids = a["ids"].to(device)
                    mask = a["mask"].to(device)
                    token_type_ids = a["token_type_ids"].to(device)
                    output = model(ids, mask, token_type_ids)
                    output = output[0].squeeze(-1)

                    preds.append(output.detach().cpu().numpy())

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
        logging.info('Load model save states.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        if path:
            pass
        else:
            path = os.getcwd()
        if self.prediction_mode:
            pthes = [os.path.join(f"{path}/", s) for s in os.listdir(f"{path}/") if ".pth" in s]
            return pthes
        else:
            pthes = [os.path.join(f"{path}/", s) for s in os.listdir(f"{path}/") if ".pth" in s]
            return pthes

    def transformer_train(self):
        if self.prediction_mode:
            pass
        else:
            logging.info('Start NLP transformer training.')
            logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
            self.reset_test_train_index()

            train_dataloader = self.create_train_dataloader()
            test_dataloader = self.create_test_dataloader()
            model, optimizer, train_steps, num_steps, scheduler = self.model_setup()
            scheduler = get_linear_schedule_with_warmup(optimizer, num_steps, train_steps)

            trainlosses = []
            vallosses = []
            bestscore = None
            trainscores = []
            validscores = []

            for epoch in tqdm(range(self.transformer_settings["epochs"])):
                print("---------------" + str(epoch) + "start-------------")
                trainloss, trainscore = self.training(train_dataloader, model, optimizer, scheduler)
                trainlosses.append(trainloss)
                trainscores.append(trainscore)
                print("trainscore is " + str(trainscore))
                preds, validloss, valscore = self.validating(test_dataloader, model)
                vallosses.append(validloss)
                validscores.append(valscore)

                print("valscore is " + str(valscore))
                if bestscore is None:
                    bestscore = valscore
                    print("Save first model")
                    state = {
                        'state_dict': model.state_dict(),
                        'optimizer_dict': optimizer.state_dict(),
                        "bestscore": bestscore
                    }
                    torch.save(state, "model0.pth")

                elif bestscore > valscore:
                    bestscore = valscore
                    print("found better point")
                    state = {
                        'state_dict': model.state_dict(),
                        'optimizer_dict': optimizer.state_dict(),
                        "bestscore": bestscore
                    }
                    torch.save(state, "model0.pth")
                else:
                    pass

            bestscores = []
            bestscores.append(bestscore)

            for fold in range(1, 5):

                self.reset_test_train_index()
                X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

                # initializing the data
                train_dataloader = self.create_train_dataloader()
                test_dataloader = self.create_test_dataloader()

                model = self.create_bert_regression_model(chosen_model=self.transformer_chosen)
                model.to(device)
                LR = 2e-5 #1e-3
                optimizer = AdamW(model.parameters(), LR, betas=(0.99, 0.999), weight_decay=1e-2)
                train_steps = int(
                    len(X_train) / self.transformer_settings["train_batch_size"] * self.transformer_settings["epochs"])
                num_steps = int(train_steps * 0.1)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_steps, train_steps)

                trainlosses = []
                vallosses = []
                bestscore = None
                trainscores = []
                validscores = []

                for epoch in tqdm(range(self.transformer_settings["epochs"])):
                    print("---------------" + str(epoch) + "start-------------")
                    trainloss, trainscore = self.training(train_dataloader, model, optimizer, scheduler)
                    trainlosses.append(trainloss)
                    trainscores.append(trainscore)

                    print("trainscore is " + str(trainscore))
                    preds, validloss, valscore = self.validating(test_dataloader, model)

                    vallosses.append(validloss)
                    validscores.append(valscore)
                    print("valscore is " + str(valscore))

                    if bestscore is None:
                        bestscore = valscore
                        print("Save first model")
                        state = {
                            'state_dict': model.state_dict(),
                            'optimizer_dict': optimizer.state_dict(),
                            "bestscore": bestscore
                        }
                        torch.save(state, "model" + str(fold) + ".pth")
                    elif bestscore > valscore:
                        bestscore = valscore
                        print("found better point")
                        state = {
                            'state_dict': model.state_dict(),
                            'optimizer_dict': optimizer.state_dict(),
                            "bestscore": bestscore
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

    def transformer_predict(self):
        logging.info('Start NLP transformer prediction.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        self.reset_test_train_index()
        model = self.create_bert_regression_model(chosen_model=self.transformer_chosen)
        pthes = self.load_model_states()
        print(pthes)
        pred_dataloader = self.pred_dataloader()
        allpreds, mode_cols = self.predicting(pred_dataloader, model, pthes)

        if self.prediction_mode:
            self.dataframe["transformers_median"] = self.dataframe[mode_cols].median(axis=1)
            self.dataframe["transformers_mean"] = self.dataframe[mode_cols].mean(axis=1)
            self.predicted_values['nlp_transformer'] = self.dataframe["transformers_mean"]
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()

            if self.transformer_settings["keep_best_model_only"]:
                # we check, if one savestate underperforms and delete him out
                scorings = []
                states = []
                for state in mode_cols:
                    state_score = self.median_abs_error_eval(Y_test, X_test[state])
                    #print(state_score)
                    scorings.append(state_score)
                    states.append(state)
                scorings_arr = np.array(scorings)
                scorings_mean = np.mean(scorings_arr)
                scorings_std = np.std(scorings_arr)
                #print(scorings_std)
                keep_state = scorings_arr < scorings_mean+scorings_std
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

            X_test["transformers_median"] = X_test[mode_cols].median(axis=1)
            X_test["transformers_mean"] = X_test[mode_cols].mean(axis=1)
            self.predicted_values['nlp_transformer'] = X_test["transformers_median"]