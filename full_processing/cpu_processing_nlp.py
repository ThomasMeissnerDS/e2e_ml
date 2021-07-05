from full_processing import cpu_preprocessing
from sklearn.decomposition import PCA
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag, pos_tag_sents
import pandas as pd
import numpy as np
import logging
import gc

# TODO: Continue on NLP
"""
THIS PART IS UNFINISHED AND UNTESTED.
"""


class NlpPreprocessing(cpu_preprocessing.PreProcessing):
    def spacy_features(self, df: pd.DataFrame, text_column):
        """
        This function generates features using spacy en_core_wb_lg
        I learned about this from these resources:
        https://www.kaggle.com/konradb/linear-baseline-with-cv
        https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
        """
        logging.info('Download spacy language package.')
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

    def pos_tagging(self):
        def dataframe_nlp(df, text_cols, mode='fit', loaded_pca=None):
            spacy_cols = []
            for text_col in text_cols:
                try:
                    df[f'nof_words_{text_col}'] = df[text_col].apply(lambda s: len(s.split(' ')))
                    if df[f'nof_words_{text_col}'].max() >= 3:
                        spacy_cols.append(text_col)
                        spacy_df = pd.DataFrame(self.spacy_features(df, text_col), columns=self.get_spacy_col_names())
                        df = pd.merge(df, spacy_df, left_index=True, right_index=True)
                        pos_df = pd.DataFrame(df[text_col].apply(lambda p: self.pos_tag_features(p)).tolist(),
                                              columns=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS",
                                                       "MD",
                                                       "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR",
                                                       "RBS", "RP", "TO", "UH",
                                                       "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"])
                        df = pd.merge(df, pos_df, left_index=True, right_index=True)
                    else:
                        pass
                except AttributeError:
                    pass
            if len(spacy_cols) > 0:
                df_temp = df[spacy_cols].copy()
                pca = PCA(n_components=2)
                if mode == 'fit':
                    comps = pca.fit_transform(df_temp)
                else:
                    pca = loaded_pca
                    comps = pca.transform(df_temp)
                pos_df = pd.DataFrame(comps, columns=['POS PC-1', 'POS PC-2'])
                for col in pos_df:
                    df[col] = pos_df[col]
                try:
                    del df_temp
                    del comps
                    _ = gc.collect()
                except Exception:
                    pass
            return df, pca

        logging.info('Start spacy POS tagging loop.')
        algorithm = 'spacy_pos'
        if self.prediction_mode:
            text_columns = self.preprocess_decisions[f"text_processing_columns"][f"{algorithm}"]
            self.dataframe, self.preprocess_decisions[f"text_processing_columns"]["pos_tagger_pca"] = dataframe_nlp(
                self.dataframe, text_columns, mode='transform',
                loaded_pca=self.preprocess_decisions[f"text_processing_columns"]["pos_tagger_pca"])
            logging.info('Finished spacy POS tagging loop.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.preprocess_decisions[f"text_processing_columns"] = {}
            text_columns = X_train.select_dtypes(include=['object']).columns
            X_train, self.preprocess_decisions[f"text_processing_columns"]["pos_tagger_pca"] = dataframe_nlp(X_train,
                                                                                                             text_columns,
                                                                                                             mode='fit')
            X_test, self.preprocess_decisions[f"text_processing_columns"]["pos_tagger_pca"] = dataframe_nlp(X_test,
                                                                                                            text_columns,
                                                                                                            mode='transform',
                                                                                                            loaded_pca=
                                                                                                            self.preprocess_decisions[
                                                                                                                f"text_processing_columns"][
                                                                                                                "pos_tagger_pca"])
            self.preprocess_decisions[f"text_processing_columns"][f"{algorithm}"] = text_columns
            logging.info('Finished spacy POS tagging loop.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
