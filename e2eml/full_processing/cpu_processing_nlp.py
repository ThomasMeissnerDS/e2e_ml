from e2eml.full_processing import cpu_preprocessing
from sklearn.decomposition import PCA
import re
import spacy
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn.linear_model import Ridge
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

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
        # clean (convert to lowercase and remove punctuations and   characters and then strip)
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())

        # Tokenize (convert from string to list)
        lst_text = text.split()    # remove Stopwords
        if lst_stopwords is not None:
            lst_text = [word for word in lst_text if word not in lst_stopwords]

        # Stemming (remove -ing, -ly, ...)
        if flg_stemm == True:
            ps = nltk.stem.porter.PorterStemmer()
            lst_text = [ps.stem(word) for word in lst_text]

        # Lemmatisation (convert the word into root word)
        if flg_lemm == True:
            lem = nltk.stem.wordnet.WordNetLemmatizer()
            lst_text = [lem.lemmatize(word) for word in lst_text]

        # back to string from list
        text = " ".join(lst_text)
        return text

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
            print(df.info())
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

    def tfidf_pca(self, df, text_cols, mode='fit', pca_pos_tags=True):
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
                        tfids = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", max_features=10000)
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
            print(df.info())
        return df

    def tfidf_vectorizer_to_pca(self, pca_pos_tags=True):
        self.get_current_timestamp(task='Start Spacy, POS tagging')
        logging.info('Start TFIDF to PCA loop.')
        algorithm = 'spacy_pos'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            self.dataframe = self.tfidf_pca(
                self.dataframe, text_columns, mode='transform', pca_pos_tags=pca_pos_tags)
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
                                     pca_pos_tags=pca_pos_tags
                                     )
            X_test = self.tfidf_pca(X_test,
                                    self.nlp_columns, mode='transform',
                                    pca_pos_tags=pca_pos_tags
                                    )
            logging.info('Finished TFIDF to PCA loop.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)

    def tfidf_naive_bayes(self, df, text_cols, target_col=None, mode='fit'):
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
                    ridge = self.preprocess_decisions[f"tfidf_bayes"][f"ridge_model_{text_col}"]
                    y_hat_ridge = ridge.predict(X_test_bayes)
                    df[f"tfid_ridge_pred_{text_col}"] = y_hat_ridge

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
                        tfids = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", max_features=10000)
                        tfids.fit(df[text_col])
                        X_train_bayes = tfids.transform(df[text_col])
                        Y_train_bayes = target_col
                        self.preprocess_decisions[f"tfidf_bayes"][f"tfidf_{text_col}"] = tfids
                        if self.class_problem == 'binary':
                            classifier = naive_bayes.MultinomialNB()
                        elif self.class_problem == 'multiclass':
                            classifier = naive_bayes.MultinomialNB()
                        elif self.class_problem == 'regression':
                            classifier = lgb.LGBMRegressor()
                            ridge = Ridge(alpha=1.0)
                            ridge.fit(X_train_bayes, Y_train_bayes)
                            y_hat_ridge = ridge.predict(X_train_bayes)
                            df[f"tfid_ridge_pred_{text_col}"] = y_hat_ridge
                            self.preprocess_decisions[f"tfidf_bayes"][f"ridge_model_{text_col}"] = ridge
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
            print(df.info())
        return df

    def tfidf_naive_bayes_proba(self, pca_pos_tags=True):
        self.get_current_timestamp(task='Start Spacy, POS tagging')
        logging.info('Start TFIDF to PCA loop.')
        algorithm = 'spacy_pos'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            self.dataframe = self.tfidf_naive_bayes(
                self.dataframe, text_columns, mode='transform')
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
                                     text_columns, mode='fit', target_col=Y_train
                                     )
            X_test = self.tfidf_naive_bayes(X_test,
                                    self.nlp_columns, mode='transform'
                                    )
            logging.info('Finished TFIDF to PCA loop.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
