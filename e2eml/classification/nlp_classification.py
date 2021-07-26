import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import AutoModel, BertTokenizerFast, AdamW
from sklearn.utils.class_weight import compute_class_weight
from e2eml.full_processing import postprocessing
# specify GPU
device = torch.device("cuda")


class ClassificationModels(postprocessing.FullPipeline):
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
    :param nlp_columns: NLP columns can be passed specifically. This only makes sense, if the chosen blueprint runs under 'nlp' processing.
    If NLP columns are not declared, categorical columns will be interpreted as such.
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
    def import_transformer_model_tokenizer(self, transformer_chosen='bert-base-uncased'):
        # import BERT-base pretrained model
        bert = AutoModel.from_pretrained(transformer_chosen)
        # Load the BERT tokenizer
        tokenizer = BertTokenizerFast.from_pretrained(transformer_chosen)
        if "nlp_transformers" in self.preprocess_decisions:
            pass
        else:
            self.preprocess_decisions[f"nlp_transformers"] = {}
        self.preprocess_decisions[f"nlp_transformers"][f"transformer_model_{transformer_chosen}"] = bert
        self.preprocess_decisions[f"nlp_transformers"][f"transformer_tokenizer_{transformer_chosen}"] = tokenizer

    def check_max_sentence_length(self, nlp_col):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        seq_len = [len(i.split()) for i in X_train[nlp_col]]
        pd.Series(seq_len).hist(bins=30)
        self.max_nlp_text_len = max(seq_len)

    def sequence_encoding(self, max_length=None, transformer_chosen='bert-base-uncased', text_columns=None):
        if max_length:
            pass
        elif not max_length and not self.max_nlp_text_len:
            max_length = 256
        elif not max_length and self.max_nlp_text_len:
            max_length = self.max_nlp_text_len

        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        if text_columns:
            pass
        elif not self.nlp_columns:
            text_columns = X_train.select_dtypes(include=['object']).columns
        elif self.nlp_columns:
            text_columns = self.nlp_columns

        for text_col in text_columns:
            tokenizer = self.preprocess_decisions[f"nlp_transformers"][f"transformer_tokenizer_{transformer_chosen}"]
            if self.prediction_mode:
                tokens = tokenizer.batch_encode_plus(
                    self.dataframe[text_col].tolist(),
                    max_length=max_length,
                    pad_to_max_length=True,
                    truncation=True
                )
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict"] = tokens
            else:
                tokens = tokenizer.batch_encode_plus(
                    X_train[text_col].tolist(),
                    max_length=max_length,
                    pad_to_max_length=True,
                    truncation=True
                )
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train"] = tokens

                tokens = tokenizer.batch_encode_plus(
                    X_test[text_col].tolist(),
                    max_length=max_length,
                    pad_to_max_length=True,
                    truncation=True
                )
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test"] = tokens


    def token_list_to_tensor(self, transformer_chosen='bert-base-uncased', text_columns=None):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        if text_columns:
            pass
        elif not self.nlp_columns:
            text_columns = X_train.select_dtypes(include=['object']).columns
        elif self.nlp_columns:
            text_columns = self.nlp_columns
        for text_col in text_columns:
            if self.prediction_mode:
                tokens = self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict"]
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_seq_predict"] = torch.tensor(tokens['input_ids'])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_mask_predict"] = torch.tensor(tokens['attention_mask'])
                #self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_y_true_predict"] = torch.tensor(Y_train) # No Y_val
            else:
                tokens = self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train"]
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_seq_train"] = torch.tensor(tokens['input_ids'])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_mask_train"] = torch.tensor(tokens['attention_mask'])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_y_true_train"] = torch.tensor(Y_train)

                tokens = self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test"]
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_seq_test"] = torch.tensor(tokens['input_ids'])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_mask_test"] = torch.tensor(tokens['attention_mask'])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_y_true_test"] = torch.tensor(Y_test)

    def batch_data_loader(self, transformer_chosen='bert-base-uncased', text_columns=None, batch_size=16):
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        if text_columns:
            pass
        elif not self.nlp_columns:
            text_columns = X_train.select_dtypes(include=['object']).columns
        elif self.nlp_columns:
            text_columns = self.nlp_columns
        for text_col in text_columns:
            if self.prediction_mode:
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict_data"] = TensorDataset(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_seq_predict"],
                                                                                                                                       self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_mask_predict"],
                                                                                                                                       self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_y_true_predict"])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict_sampler"] = RandomSampler(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict_data"])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict_dataloader"] = DataLoader(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict_data"],
                                                                                                                                          sampler=self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_predict_sampler"],
                                                                                                                                          batch_size=batch_size)
            else:
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train_data"] = TensorDataset(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_seq_train"],
                                                                                                                                        self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_mask_train"],
                                                                                                                                        self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_y_true_train"])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train_sampler"] = RandomSampler(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train_data"])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train_dataloader"] = DataLoader(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train_data"],
                                                                                                                                           sampler=self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_train_sampler"],
                                                                                                                                           batch_size=batch_size)

                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test_data"] = TensorDataset(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_seq_test"],
                                                                                                                                        self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_mask_test"],
                                                                                                                                        self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_token_y_true_test"])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test_sampler"] = RandomSampler(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test_data"])
                self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test_dataloader"] = DataLoader(self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test_data"],
                                                                                                                                           sampler=self.preprocess_decisions[f"nlp_transformers"][f"tokenized_{text_col}_{transformer_chosen}_test_sampler"],
                                                                                                                                           batch_size=batch_size)

    def freeze_parameters(self, transformer_chosen='bert-base-uncased'):
        # freeze all the parameters
        for param in self.preprocess_decisions[f"nlp_transformers"][f"transformer_model_{transformer_chosen}"].parameters():
            param.requires_grad = False


class BERTArch(nn.Module, ClassificationModels):
    def __init__(self, bert):
        super(BERTArch, self).__init__()
        self.bert = bert
        # dropout layer
        self.dropout = nn.Dropout(0.1)
        # relu activation function
        self.relu = nn.ReLU()
        # dense layer 1
        self.fc1 = nn.Linear(768,512)
        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512,2)
        #softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):
        #pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)

        return x

class BERTArchModel(BERTArch):
    def define_tansformer_model(self, transformer_chosen='bert-base-uncased')
        X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
        model = BERTArch(self.preprocess_decisions[f"nlp_transformers"][f"transformer_model_{transformer_chosen}"])
        # push the model to GPU
        model = model.to(device)
        # define the optimizer
        optimizer = AdamW(model.parameters(), lr = 1e-5)
        #compute the class weights
        class_weights = compute_class_weight('balanced', np.unique(Y_train), Y_train)
        # converting list of class weights to a tensor
        weights= torch.tensor(class_weights,dtype=torch.float)

        # push to GPU
        weights = weights.to(device)

        # define the loss function
        cross_entropy  = nn.NLLLoss(weight=weights)

        # number of training epochs
        epochs = 10

