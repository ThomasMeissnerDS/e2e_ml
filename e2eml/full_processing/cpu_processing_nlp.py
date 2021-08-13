from e2eml.full_processing import cpu_preprocessing
from sklearn.decomposition import PCA
import re
import string
import spacy
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag
from textblob import TextBlob
import transformers
from transformers import AutoModel, AutoTokenizer, AdamW, BertModel, RobertaModel, RobertaTokenizer, BertTokenizer, ElectraTokenizer, ElectraForSequenceClassification, XLNetForSequenceClassification, XLMRobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.linear_model import Ridge, ElasticNet
from imblearn.over_sampling import SMOTE
from vowpalwabbit.sklearn_vw import VWClassifier, VWRegressor
import lightgbm as lgb
import pandas as pd
import numpy as np
import random
import logging
import psutil

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# TODO: Continue on NLP
"""
THIS PART IS UNFINISHED AND UNTESTED.
"""


class NlpPreprocessing(cpu_preprocessing.PreProcessing):
    """
    Preprocess a string.
    :parameter
        :param text: string - name of column containing text
        :param lst_stopwords: list - list of stopwords to remove
        :param flg_stemm: bool - whether stemming is to be applied
        :param flg_lemm: bool - whether lemmitisation is to be applied
    :return
        cleaned text
    """
    def utils_preprocess_text(self, text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
        logging.info('Start text cleaning.')
        logging.info(f'RAM memory {psutil.virtual_memory()[2]} percent used.')
        # clean (convert to lowercase and remove punctuations and   characters and then strip)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

        # Tokenize (convert from string to list)
        lst_text = text.split()    # remove Stopwords
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in lst_stopwords]

        # Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            logging.info('Start text stemming.')
            ps = nltk.stem.porter.PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]

        # Lemmatisation (convert the word into root word)
        if flg_lemm == True:
            logging.info('Start text lemmatisation.')
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]

        # back to string from list
        text = " ".join(lst_text)
        return text

    def remove_url(self, text):
        logging.info('Start removing URLs.')
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r'', str(text))

    def remove_punct(self, text):
        logging.info('Start removing punctuation.')
        table = str.maketrans('', '', string.punctuation)
        return text.translate(table)

    def remove_html(self, text):
        logging.info('Start removing HTML.')
        html = re.compile(r'<.*?>')
        return html.sub(r'', str(text))

    def remove_emoji(self, text):
        logging.info('Start removing emojis.')
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', str(text))

    def decontraction(self, text):
        logging.info('Start decontraction.')
        text = re.sub(r"won\'t", " will not", text)
        text = re.sub(r"won\'t've", " will not have", text)
        text = re.sub(r"can\'t", " can not", text)
        text = re.sub(r"don\'t", " do not", text)

        text = re.sub(r"can\'t've", " can not have", text)
        text = re.sub(r"ma\'am", " madam", text)
        text = re.sub(r"let\'s", " let us", text)
        text = re.sub(r"ain\'t", " am not", text)
        text = re.sub(r"shan\'t", " shall not", text)
        text = re.sub(r"sha\n't", " shall not", text)
        text = re.sub(r"o\'clock", " of the clock", text)
        text = re.sub(r"y\'all", " you all", text)

        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"n\'t've", " not have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'d've", " would have", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ll've", " will have", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        text = re.sub(r"\'re", " are", text)
        return str(text)

    def seperate_alphanumeric(self, text):
        logging.info('Start seperating alphanumeric.')
        words = text
        words = re.findall(r"[^\W\d_]+|\d+", words)
        return " ".join(words)

    def cont_rep_char(self, text):
        tchr = text.group(0)

        if len(tchr) > 1:
            return tchr[0:2]

    def unique_char(self, rep, text):
        substitute = re.sub(r'(\w)\1+', rep, text)
        return str(substitute)

    def regex_clean_text_data(self):
        logging.info('Start text cleaning.')
        self.get_current_timestamp(task='Start text cleaning.')
        logging.info('Start text cleaning.')
        algorithm = 'regex_text'
        if self.prediction_mode:
            text_columns = []
            text_columns.append(self.nlp_transformer_columns)
            for text_col in text_columns:
                self.dataframe[text_col] = self.dataframe[text_col].apply(lambda x : self.remove_url(x))
                self.dataframe[text_col] = self.dataframe[text_col].apply(lambda x : self.remove_punct(x))
                self.dataframe[text_col] = self.dataframe[text_col].apply(lambda x : self.remove_emoji(x))
                self.dataframe[text_col] = self.dataframe[text_col].apply(lambda x : self.decontraction(x))
                self.dataframe[text_col] = self.dataframe[text_col].apply(lambda x : self.seperate_alphanumeric(x))
                self.dataframe[text_col] = self.dataframe[text_col].apply(lambda x : self.unique_char(self.cont_rep_char,x))
            logging.info('Finished text cleaning.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            text_columns = []
            text_columns.append(self.nlp_transformer_columns)

            for text_col in text_columns:
                try:
                    X_train[text_col] = X_train[text_col].apply(lambda x : self.remove_url(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_train[text_col] = X_train[text_col].apply(lambda x : self.remove_punct(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_train[text_col] = X_train[text_col].apply(lambda x : self.remove_emoji(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_train[text_col] = X_train[text_col].apply(lambda x : self.decontraction(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_train[text_col] = X_train[text_col].apply(lambda x : self.seperate_alphanumeric(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_train[text_col] = X_train[text_col].apply(lambda x : self.unique_char(self.cont_rep_char,x))
                except (TypeError, AttributeError):
                    pass

                try:
                    X_test[text_col] = X_test[text_col].apply(lambda x : self.remove_url(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_test[text_col] = X_test[text_col].apply(lambda x : self.remove_punct(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_test[text_col] = X_test[text_col].apply(lambda x : self.remove_emoji(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_test[text_col] = X_test[text_col].apply(lambda x : self.decontraction(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_test[text_col] = X_test[text_col].apply(lambda x : self.seperate_alphanumeric(x))
                except (TypeError, AttributeError):
                    pass
                try:
                    X_test[text_col] = X_test[text_col].apply(lambda x : self.unique_char(self.cont_rep_char,x))
                except (TypeError, AttributeError):
                    pass
            logging.info('Finished text cleaning.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def textBlob_sentiment_polarity_score(self, text):
        # This polarity score is between -1 to 1
        polarity = TextBlob(text).sentiment.polarity
        return polarity

    def append_text_sentiment_score(self, text_columns=None):
        self.get_current_timestamp(task='Start text cleaning.')
        logging.info('Start sentiment polarity score.')
        algorithm = 'textblob_sentiment_score'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            for text_col in text_columns:
                self.dataframe[f"{algorithm}_{text_col}"] = self.dataframe[text_col].apply(lambda x : self.textBlob_sentiment_polarity_score(x))
            logging.info('Finished text cleaning.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            text_columns = self.nlp_columns
            for text_col in text_columns:
                X_train[f"{algorithm}_{text_col}"] = X_train[text_col].apply(lambda x : self.textBlob_sentiment_polarity_score(x))
                X_test[f"{algorithm}_{text_col}"] = X_test[text_col].apply(lambda x : self.textBlob_sentiment_polarity_score(x))
            logging.info('Finished text cleaning.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def spacy_features(self, df: pd.DataFrame, text_column):
        """
        This function generates features using spacy en_core_wb_lg
        More information:
        https://www.kaggle.com/konradb/linear-baseline-with-cv
        https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
        """
        logging.info('Download spacy language package.')
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print('Downloading language model for the spaCy POS tagger\n'
                  "(don't worry, this will only happen once)")
            from spacy.cli import download
            download('en')
            nlp = spacy.load('en_core_web_sm')
            # https://spacy.io/universe/project/spacy-transformers
            # https://spacy.io/models
            #!python -m spacy download en_core_web_trf
            #!pip install spacy-transformers
            # nlp = spacy.load('en_core_web_trf')
        # nlp = spacy.load('en_core_web_lg')
        with nlp.disable_pipes():
            vectors = np.array([nlp(text).vector for text in df[text_column]])
        return vectors

    def get_spacy_col_names(self):
        logging.info('Get spacy column names.')
        names = list()
        for i in range(96):  # TODO: Why 96?
            names.append(f"spacy_{i}")
        return names

    def pos_tag_features(self, passage: str):
        """
        This function counts the number of times different parts of speech occur in an excerpt. POS tags are listed here:
        https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        """
        logging.info('Get POS tags.')
        pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
                    "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                    "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]

        tags = pos_tag(word_tokenize(passage))
        tag_list = list()
        for tag in pos_tags:
            tag_list.append(len([i[0] for i in tags if i[1] == tag]))
        return tag_list

    def pos_tagging_pca_nlp_cols(self, df, text_cols, mode='fit', pca_pos_tags=True):
        if mode == 'transform':
            pass
        else:
            pos_columns = set([])
        target_columns = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
                          "MD",
                          "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR",
                          "RBS", "RP", "TO", "UH",
                          "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]
        if mode == 'transform':
            for text_col in text_cols:
                temp_target_columns = [x + text_col for x in target_columns]
                df[text_col].fillna('None', inplace=True)
                spacy_df = pd.DataFrame(self.spacy_features(df, text_col), columns=self.get_spacy_col_names())
                temp_df = pd.merge(df, spacy_df, left_index=True, right_index=True, how='left')
                pos_df = pd.DataFrame(temp_df[text_col].apply(lambda p: self.pos_tag_features(p)).tolist(),
                                      columns=temp_target_columns)
                pos_df.fillna(0, inplace=True)
                if pca_pos_tags:
                    self.get_current_timestamp(task='PCA POS tags')
                    logging.info('Start to PCA POS tags.')
                    pca = self.preprocess_decisions[f"spacy_pos"][f"pos_pca_{text_col}"]
                    comps = pca.transform(pos_df)
                    pos_pca_cols = [f'POS PC-1 {text_col}', f'POS PC-2 {text_col}']
                    pos_df = pd.DataFrame(comps, columns=pos_pca_cols)
                    pos_df_pca = pos_df[pos_pca_cols]
                    df = pd.merge(df, pos_df_pca, left_index=True, right_index=True, how='left')
                else:
                    df = pd.merge(df, pos_df, left_index=True, right_index=True, how='left')
        elif mode == 'fit':
            # if self.nlp_columns
            nlp_columns = []
            for text_col in text_cols:
                try:
                    # do we have at least 3 words?
                    df[f'nof_words_{text_col}'] = df[text_col].apply(lambda s: len(s.split(' ')))
                    if df[f'nof_words_{text_col}'].max() >= 3:
                        temp_target_columns = [x + text_col for x in target_columns]
                        df[text_col].fillna('None', inplace=True)
                        spacy_df = pd.DataFrame(self.spacy_features(df, text_col), columns=self.get_spacy_col_names())
                        temp_df = pd.merge(df, spacy_df, left_index=True, right_index=True, how='left')
                        pos_df = pd.DataFrame(temp_df[text_col].apply(lambda p: self.pos_tag_features(p)).tolist(),
                                              columns=temp_target_columns)
                        pos_df.fillna(0, inplace=True)
                        for col in pos_df.columns:
                            pos_columns.add(col)
                        if pca_pos_tags:
                            self.get_current_timestamp(task='PCA POS tags')
                            logging.info('Start to PCA POS tags.')
                            pca = PCA(n_components=2)
                            comps = pca.fit_transform(pos_df)
                            self.preprocess_decisions[f"spacy_pos"][f"pos_pca_{text_col}"] = pca
                            pos_pca_cols = [f'POS PC-1 {text_col}', f'POS PC-2 {text_col}']
                            pos_df = pd.DataFrame(comps, columns=pos_pca_cols)
                            pos_df_pca = pos_df[pos_pca_cols]
                            df = pd.merge(df, pos_df_pca, left_index=True, right_index=True, how='left')
                        else:
                            df = pd.merge(df, pos_df, left_index=True, right_index=True, how='left')
                        nlp_columns.append(text_col)
                    else:
                        pass
                    df.drop(f'nof_words_{text_col}', axis=1, inplace=True)
                except AttributeError:
                    pass
            if pca_pos_tags:
                # get unique pos tag columns
                unique_pos_cols = list(set(pos_columns))
                self.preprocess_decisions[f"spacy_pos"]["pos_tagger_cols"] = unique_pos_cols
            # get unique original column names
            unique_nlp_cols = list(set(nlp_columns))
            self.nlp_columns = unique_nlp_cols
        return df

    def pos_tagging_pca(self, pca_pos_tags=True):
        self.get_current_timestamp(task='Start Spacy, POS tagging')
        logging.info('Start spacy POS tagging loop.')
        algorithm = 'spacy_pos'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            self.dataframe = self.pos_tagging_pca_nlp_cols(
                self.dataframe, text_columns, mode='transform', pca_pos_tags=pca_pos_tags)
            logging.info('Finished spacy POS tagging loop.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if pca_pos_tags:
                self.preprocess_decisions[f"spacy_pos"] = {}
            else:
                pass
            if not self.nlp_columns:
                text_columns = X_train.select_dtypes(include=['object']).columns
            else:
                text_columns = self.nlp_columns
            X_train = self.pos_tagging_pca_nlp_cols(X_train,
                                                    text_columns, mode='fit',
                                                    pca_pos_tags=pca_pos_tags
                                                    )
            X_test = self.pos_tagging_pca_nlp_cols(X_test,
                                                   self.nlp_columns, mode='transform',
                                                   pca_pos_tags=pca_pos_tags
                                                   )
            logging.info('Finished spacy POS tagging loop.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def tfidf_pca(self, df, text_cols, mode='fit', pca_pos_tags=True, ngram_range=(1, 2)):
        if mode == 'transform':
            pass
        else:
            pass
        if mode == 'transform':
            for text_col in text_cols:
                df[text_col].fillna('None', inplace=True)
                tfids = self.preprocess_decisions[f"tfidf_vectorizer"][f"tfidf_{text_col}"]
                vector = list(tfids.transform(df[text_col]).toarray())
                if pca_pos_tags:
                    self.get_current_timestamp(task='PCA POS tags')
                    logging.info('Start to PCA POS tags.')
                    pca = self.preprocess_decisions[f"tfidf_vectorizer"][f"tfidf_pca_{text_col}"]
                    comps = pca.transform(vector)
                    tfidf_pca_cols = [f'TFIDF PC-1 {text_col}', f'TFIDF PC-2 {text_col}']
                    pos_df = pd.DataFrame(comps, columns=tfidf_pca_cols)
                    tfidf_df_pca = pos_df[tfidf_pca_cols]
                    df = pd.merge(df, tfidf_df_pca, left_index=True, right_index=True, how='left')
                else:
                    pass
        elif mode == 'fit':
            # if self.nlp_columns
            nlp_columns = []
            for text_col in text_cols:
                try:
                    # do we have at least 3 words?
                    df[f'nof_words_{text_col}'] = df[text_col].apply(lambda s: len(s.split(' ')))
                    if df[f'nof_words_{text_col}'].max() >= 3:
                        df[text_col].fillna('None', inplace=True)
                        tfids = TfidfVectorizer(ngram_range=ngram_range, strip_accents="unicode", max_features=10000)
                        vector = list(tfids.fit_transform(df[text_col]).toarray())
                        self.preprocess_decisions[f"tfidf_vectorizer"][f"tfidf_{text_col}"] = tfids
                        if pca_pos_tags:
                            self.get_current_timestamp(task='PCA TfIDF matrix')
                            logging.info('Start to PCA TfIDF matrix.')
                            pca = PCA(n_components=2)
                            comps = pca.fit_transform(vector)
                            self.preprocess_decisions[f"tfidf_vectorizer"][f"tfidf_pca_{text_col}"] = pca
                            tfidf_pca_cols = [f'TFIDF PC-1 {text_col}', f'TFIDF PC-2 {text_col}']
                            pos_df = pd.DataFrame(comps, columns=tfidf_pca_cols)
                            tfidf_df_pca = pos_df[tfidf_pca_cols]
                            df = pd.merge(df, tfidf_df_pca, left_index=True, right_index=True, how='left')
                        else:
                            pass
                        nlp_columns.append(text_col)
                    else:
                        pass
                    df.drop(f'nof_words_{text_col}', axis=1, inplace=True)
                except AttributeError:
                    pass
            # get unique original column names
            unique_nlp_cols = list(set(nlp_columns))
            self.nlp_columns = unique_nlp_cols
        return df

    def tfidf_vectorizer_to_pca(self, pca_pos_tags=True, ngram_range=(1, 2)):
        self.get_current_timestamp(task='Start Spacy, POS tagging')
        logging.info('Start TFIDF to PCA loop.')
        algorithm = 'spacy_pos'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            self.dataframe = self.tfidf_pca(
                self.dataframe, text_columns, mode='transform', pca_pos_tags=pca_pos_tags, ngram_range=ngram_range)
            logging.info('Finished TFIDF to PCA loop.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if pca_pos_tags:
                self.preprocess_decisions[f"tfidf_vectorizer"] = {}
            else:
                pass
            if not self.nlp_columns:
                text_columns = X_train.select_dtypes(include=['object']).columns
            else:
                text_columns = self.nlp_columns
            X_train = self.tfidf_pca(X_train,
                                     text_columns, mode='fit',
                                     pca_pos_tags=pca_pos_tags,
                                     ngram_range=ngram_range
                                     )
            X_test = self.tfidf_pca(X_test,
                                    self.nlp_columns, mode='transform',
                                    pca_pos_tags=pca_pos_tags,
                                    ngram_range=ngram_range
                                    )
            logging.info('Finished TFIDF to PCA loop.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def tfidf_naive_bayes(self, df, text_cols, target_col=None, mode='fit', analyzer="char_wb", ngram_range=(1, 2)):
        if mode == 'transform':
            pass
        else:
            pass
        if mode == 'transform':
            for text_col in text_cols:
                lst_stopwords = nltk.corpus.stopwords.words("english")
                df[text_col] = df[text_col].apply(lambda x:
                                                  self.utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,
                                                                             lst_stopwords=lst_stopwords))
                df[text_col].fillna('None', inplace=True)
                tfids = self.preprocess_decisions[f"tfidf_bayes"][f"tfidf_{text_col}"]
                classifier = self.preprocess_decisions[f"tfidf_bayes"][f"bayes_model_{text_col}"]
                X_test_bayes = tfids.transform(df[text_col])
                y_hat_bayes = classifier.predict(X_test_bayes)
                df[f"tfid_bayes_pred_{text_col}"] = y_hat_bayes
                if self.class_problem == 'regression':
                    # ridge
                    ridge = self.preprocess_decisions[f"tfidf_bayes"][f"ridge_model_{text_col}"]
                    y_hat_ridge = ridge.predict(X_test_bayes)
                    df[f"tfid_ridge_pred_{text_col}"] = y_hat_ridge
                    # elasticnet
                    elast = self.preprocess_decisions[f"tfidf_bayes"][f"elast_model_{text_col}"]
                    y_hat_elast = elast.predict(X_test_bayes)
                    df[f"tfid_elast_pred_{text_col}"] = y_hat_elast
                    # Vowpal Wabbit
                    vowpal = self.preprocess_decisions[f"tfidf_bayes"][f"vowpal_model_{text_col}"]
                    y_hat_vowpal = vowpal.predict(X_test_bayes)
                    df[f"tfid_vowpal_pred_{text_col}"] = y_hat_vowpal

        elif mode == 'fit':
            # if self.nlp_columns
            nlp_columns = []
            for text_col in text_cols:
                try:
                    # do we have at least 3 words?
                    df[f'nof_words_{text_col}'] = df[text_col].apply(lambda s: len(s.split(' ')))
                    if df[f'nof_words_{text_col}'].max() >= 3:
                        lst_stopwords = nltk.corpus.stopwords.words("english")
                        df[text_col] = df[text_col].apply(lambda x:
                                                              self.utils_preprocess_text(x, flg_stemm=False, flg_lemm=True,
                                                                                    lst_stopwords=lst_stopwords))
                        df[text_col].fillna('None', inplace=True)
                        tfids = TfidfVectorizer(ngram_range=ngram_range, strip_accents="unicode", max_features=20000,
                                                analyzer=analyzer)
                        tfids.fit(df[text_col])
                        X_train_bayes = tfids.transform(df[text_col])
                        Y_train_bayes = target_col
                        self.preprocess_decisions[f"tfidf_bayes"][f"tfidf_{text_col}"] = tfids
                        if self.class_problem == 'binary':
                            classifier = naive_bayes.MultinomialNB()
                        elif self.class_problem == 'multiclass':
                            classifier = naive_bayes.MultinomialNB()
                        elif self.class_problem == 'regression':
                            # lgbm
                            classifier = lgb.LGBMRegressor()
                            # Ridge
                            ridge = Ridge(alpha=1.0)
                            ridge.fit(X_train_bayes, Y_train_bayes)
                            y_hat_ridge = ridge.predict(X_train_bayes)
                            df[f"tfid_ridge_pred_{text_col}"] = y_hat_ridge
                            self.preprocess_decisions[f"tfidf_bayes"][f"ridge_model_{text_col}"] = ridge
                            # elasticnet
                            elast = ElasticNet(random_state=0)
                            elast.fit(X_train_bayes, Y_train_bayes)
                            y_hat_elast = elast.predict(X_train_bayes)
                            df[f"tfid_elast_pred_{text_col}"] = y_hat_elast
                            self.preprocess_decisions[f"tfidf_bayes"][f"elast_model_{text_col}"] = elast
                            # Vowpal Wabbit
                            vowpal = VWRegressor(convert_labels=False)
                            vowpal.fit(X_train_bayes, Y_train_bayes)
                            y_hat_vowpal = vowpal.predict(X_train_bayes)
                            df[f"tfid_vowpal_pred_{text_col}"] = y_hat_vowpal
                            self.preprocess_decisions[f"tfidf_bayes"][f"vowpal_model_{text_col}"] = vowpal
                        else:
                            classifier = naive_bayes.GaussianNB()
                        classifier.fit(X_train_bayes, Y_train_bayes)
                        y_hat_bayes = classifier.predict(X_train_bayes)
                        df[f"tfid_bayes_pred_{text_col}"] = y_hat_bayes

                        self.preprocess_decisions[f"tfidf_bayes"][f"bayes_model_{text_col}"] = classifier
                        nlp_columns.append(text_col)
                    else:
                        pass
                    df.drop(f'nof_words_{text_col}', axis=1, inplace=True)
                except AttributeError:
                    pass
            # get unique original column names
            unique_nlp_cols = list(set(nlp_columns))
            self.nlp_columns = unique_nlp_cols
        return df

    def tfidf_naive_bayes_proba(self, pca_pos_tags=True, analyzer="char_wb", ngram_range=(1, 2)):
        self.get_current_timestamp(task='Start Spacy, POS tagging')
        logging.info('Start TFIDF to PCA loop.')
        algorithm = 'spacy_pos'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            self.dataframe = self.tfidf_naive_bayes(
                self.dataframe, text_columns, mode='transform', analyzer=analyzer, ngram_range=ngram_range)
            logging.info('Finished TFIDF to PCA loop.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if pca_pos_tags:
                self.preprocess_decisions[f"tfidf_bayes"] = {}
            else:
                pass
            if not self.nlp_columns:
                text_columns = X_train.select_dtypes(include=['object']).columns
            else:
                text_columns = self.nlp_columns
            X_train = self.tfidf_naive_bayes(X_train,
                                     text_columns, mode='fit', target_col=Y_train, analyzer=analyzer, ngram_range=ngram_range
                                     )
            X_test = self.tfidf_naive_bayes(X_test,
                                    self.nlp_columns, mode='transform', analyzer=analyzer, ngram_range=ngram_range
                                    )
            logging.info('Finished TFIDF to PCA loop.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def vowpal_wabbit_text_prediction_as_feature(self, df, text_cols, target_col=None, mode='fit'):
        if mode == 'transform':
            pass
        else:
            pass
        if mode == 'transform':
            for text_col in text_cols:
                df[text_col].fillna('None', inplace=True)
                # Vowpal Wabbit
                vowpal = self.preprocess_decisions[f"tfidf_bayes"][f"vowpal_model_{text_col}"]
                y_hat_vowpal = vowpal.predict(df[text_col])
                df[f"vowpal_nlp_pred_{text_col}"] = y_hat_vowpal

        elif mode == 'fit':
            # if self.nlp_columns
            nlp_columns = []
            for text_col in text_cols:
                try:
                    # do we have at least 3 words?
                    df[f'nof_words_{text_col}'] = df[text_col].apply(lambda s: len(s.split(' ')))
                    if df[f'nof_words_{text_col}'].max() >= 3:
                        df[text_col].fillna('None', inplace=True)
                        if self.class_problem == 'binary' or self.class_problem == 'multiclass':
                            vowpal = VWClassifier(convert_labels=False)
                            vowpal.fit(df[text_col], target_col)
                            y_hat_vowpal = vowpal.predict(df[text_col])
                            df[f"vowpal_nlp_pred_{text_col}"] = y_hat_vowpal
                            self.preprocess_decisions[f"vowpal_nlp"][f"vowpal_model_{text_col}"] = vowpal
                        elif self.class_problem == 'regression':
                            vowpal = VWRegressor(convert_labels=False)
                            vowpal.fit(df[text_col], target_col)
                            y_hat_vowpal = vowpal.predict(df[text_col])
                            df[f"vowpal_nlp_pred_{text_col}"] = y_hat_vowpal
                            self.preprocess_decisions[f"vowpal_nlp"][f"vowpal_model_{text_col}"] = vowpal
                        nlp_columns.append(text_col)
                    else:
                        pass
                    df.drop(f'nof_words_{text_col}', axis=1, inplace=True)
                except AttributeError:
                    pass
            # get unique original column names
            unique_nlp_cols = list(set(nlp_columns))
            self.nlp_columns = unique_nlp_cols
        return df

    def add_vowpal_wabbit_preds_for_nlp_as_feature(self, pca_pos_tags=True):
        self.get_current_timestamp(task='Start Spacy, POS tagging')
        logging.info('Start Vowpal Wabbit NLP prediction as a feature.')
        algorithm = 'vowpal_nlp'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            self.dataframe = self.tfidf_naive_bayes(
                self.dataframe, text_columns, mode='transform')
            logging.info('Finished Vowpal Wabbit NLP prediction as a feature.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            if pca_pos_tags:
                self.preprocess_decisions[f"vowpal_nlp"] = {}
            else:
                pass
            if not self.nlp_columns:
                text_columns = X_train.select_dtypes(include=['object']).columns
            else:
                text_columns = self.nlp_columns
            X_train = self.tfidf_naive_bayes(X_train,
                                             text_columns, mode='fit', target_col=Y_train
                                             )
            X_test = self.tfidf_naive_bayes(X_test,
                                            self.nlp_columns, mode='transform'
                                            )
            logging.info('Finished Vowpal Wabbit NLP prediction as a feature.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def tfidf_smote_text_columns(self):
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            tfids = TfidfVectorizer(strip_accents="unicode", max_features=20000)
            tfids.fit(X_train[self.nlp_transformer_columns])
            vectors = tfids.transform(X_train[self.nlp_transformer_columns])
            vectors_df = pd.DataFrame(vectors.todense(), columns=tfids.get_feature_names())
            smt = SMOTE(random_state=777, k_neighbors=1)
            X_SMOTE, y_SMOTE = smt.fit_sample(vectors_df, Y_train)
            X_train = pd.DataFrame(X_SMOTE.todense(), columns=tfids.get_feature_names())
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def get_synonyms(self, word):
        """
        Get synonyms of a word
        """
        synonyms = set()

        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)

        if word in synonyms:
            synonyms.remove(word)

        return list(synonyms)

    def synonym_replacement(self, words, n):
        words = words.split()
        new_words = words.copy()
        stop_words = nltk.corpus.stopwords.words("english")
        random_word_list = list(set([word for word in words if word not in stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0

        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)

            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1

            if num_replaced >= n: #only replace up to n words
                break
        sentence = ' '.join(new_words)
        return sentence

    def replace_synonyms_to_df_copy(self, words_to_replace=3, mode ='auto'):
        """
        The function copies the original dataframe and randomly exchanges words by their synonyms. It concats the original to the
        modified dataframe and returns the new training dataset.
        :param words_to_replace:
        :return:
        """
        if self.prediction_mode:
            pass
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            X_train[self.target_variable] = Y_train
            if mode == 'auto':
                sentence_length = X_train[self.nlp_transformer_columns].apply(lambda x: np.max([len(w) for w in x.split()]))
                words_to_replace = int(sentence_length.max()/10) # exchange ten percent of the words
            train_copy = X_train.copy()
            train_copy[self.nlp_transformer_columns] = train_copy[self.nlp_transformer_columns].apply(lambda x : self.synonym_replacement(x, words_to_replace))
            X_train = pd.concat([X_train, train_copy], ignore_index=True)
            X_train = X_train.reset_index(drop=True)
            Y_train = X_train[self.target_variable]
            X_train.drop(self.target_variable, axis=1)
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def create_bert_classification_model(self, chosen_model='bert-base-uncased'):
        logging.info('Start creating or loading transformer model for classification.')
        if not self.transformer_chosen:
            chosen_model = chosen_model
        if chosen_model in ['bert-base-uncased', 'bert-base-cased', 'bert-large-cased', 'bert-base-multilingual-cased']:
            model = transformers.BertForSequenceClassification.from_pretrained(self.transformer_chosen, num_labels=self.num_classes)
        elif chosen_model in ['roberta-base', 'roberta-large', 'distilroberta-base', 'roberta-base-openai-detector']:
            model = transformers.RobertaForSequenceClassification.from_pretrained(
                self.transformer_chosen, num_labels=self.num_classes)
        elif chosen_model in ['xlm-mlm-enfr-1024', 'xlm-mlm-17-1280', 'xlm-mlm-100-1280']:
            model = transformers.XLMForSequenceClassification.from_pretrained(
                self.transformer_chosen, num_labels=self.num_classes)
        elif chosen_model in ['xlm-roberta', 'xlm-roberta-base', 'xlm-roberta-large']:
            model = transformers.XLMRobertaForSequenceClassification.from_pretrained(
                self.transformer_chosen, num_labels=self.num_classes)
        elif chosen_model == 'google/electra-small-discriminator':
            model = transformers.ElectraForSequenceClassification.from_pretrained(self.transformer_chosen, num_labels=self.num_classes)
        else:
            model = transformers.AutoModel.from_pretrained(self.transformer_chosen, num_labels=self.num_classes)
        return model

    def create_bert_regression_model(self, chosen_model='bert-base-uncased'):
        logging.info('Start creating or loading transformer model for regression.')
        if not self.transformer_chosen:
            chosen_model = chosen_model
        if chosen_model in ['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased', 'bert-base-multilingual-cased']:
            model = transformers.BertForSequenceClassification.from_pretrained(self.transformer_chosen, num_labels=1)
        elif chosen_model in ['roberta-base', 'roberta-large', 'distilroberta-base']:
            model = transformers.RobertaForSequenceClassification.from_pretrained(
                self.transformer_chosen, num_labels=1)
        elif chosen_model in ['xlm-mlm-enfr-1024', 'xlm-mlm-17-1280', 'xlm-mlm-100-1280']:
            model = transformers.XLMForSequenceClassification.from_pretrained(
                self.transformer_chosen, num_labels=1)
        elif chosen_model in ['xlm-roberta', 'xlm-roberta-base', 'xlm-roberta-large']:
            model = transformers.XLMRobertaForSequenceClassification.from_pretrained(
                self.transformer_chosen, num_labels=1)
        elif chosen_model == 'google/electra-small-discriminator':
            model = transformers.ElectraForSequenceClassification.from_pretrained(self.transformer_chosen, num_labels=1)
        else:
            model = transformers.AutoModel.from_pretrained(self.transformer_chosen, num_labels=1)
        return model

    def import_transformer_model_tokenizer(self, transformer_chosen=None):
        logging.info('Start importing transformer tokenizer.')
        if not transformer_chosen:
            transformer_chosen = 'bert-base-uncased'
        else:
            transformer_chosen = self.transformer_chosen

        if self.transformer_model_load_from_path:
            bert = AutoModel.from_pretrained(f"{self.transformer_model_load_from_path}",
                                             output_attentions=False,  # Whether the model returns attentions weights.
                                             output_hidden_states=False)
            tokenizer = transformers.AutoTokenizer.from_pretrained(f"{self.transformer_model_load_from_path}")
        else:
            if self.class_problem in ['binary', 'multiclass']:
                # import BERT-base pretrained model
                bert = self.create_bert_classification_model(transformer_chosen)
            elif self.class_problem == 'regression':
                bert = self.create_bert_regression_model(transformer_chosen)
            else:
                print("No correct ml_task defined during class instantiation.")
            # Load the BERT tokenizer
            tokenizer = AutoTokenizer.from_pretrained(transformer_chosen)
        if "nlp_transformers" in self.preprocess_decisions:
            pass
        else:
            self.preprocess_decisions[f"nlp_transformers"] = {}

        self.preprocess_decisions[f"nlp_transformers"][f"transformer_model_{transformer_chosen}"] = bert
        self.preprocess_decisions[f"nlp_transformers"][f"transformer_tokenizer_{transformer_chosen}"] = tokenizer
