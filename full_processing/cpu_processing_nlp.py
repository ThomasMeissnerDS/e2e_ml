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

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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

    def dataframe_nlp(self, df, text_cols, mode='fit'):
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
                df[text_col].fillna('None', inplace=True)
                spacy_df = pd.DataFrame(self.spacy_features(df, text_col), columns=self.get_spacy_col_names())
                temp_df = pd.merge(df, spacy_df, left_index=True, right_index=True, how='left')
                pos_df = pd.DataFrame(temp_df[text_col].apply(lambda p: self.pos_tag_features(p)).tolist(),
                                      columns=target_columns)
                pos_df.fillna(0, inplace=True)
                pca = self.preprocess_decisions[f"spacy_pos"][f"pos_pca_{text_col}"]
                comps = pca.transform(pos_df)
                pos_pca_cols = [f'POS PC-1 {text_col}', f'POS PC-2 {text_col}']
                pos_df = pd.DataFrame(comps, columns=pos_pca_cols)
                pos_df_pca = pos_df[pos_pca_cols]
                df = pd.merge(df, pos_df_pca, left_index=True, right_index=True, how='left')
        elif mode == 'fit':
            nlp_columns = []
            for text_col in text_cols:
                try:
                    # do we have at least 3 words?
                    df[f'nof_words_{text_col}'] = df[text_col].apply(lambda s: len(s.split(' ')))
                    if df[f'nof_words_{text_col}'].max() >= 3:
                        df[text_col].fillna('None', inplace=True)
                        spacy_df = pd.DataFrame(self.spacy_features(df, text_col), columns=self.get_spacy_col_names())
                        temp_df = pd.merge(df, spacy_df, left_index=True, right_index=True, how='left')
                        pos_df = pd.DataFrame(temp_df[text_col].apply(lambda p: self.pos_tag_features(p)).tolist(),
                                              columns=target_columns)
                        pos_df.fillna(0, inplace=True)
                        for col in pos_df.columns:
                            pos_columns.add(col)
                        pca = PCA(n_components=2)
                        comps = pca.fit_transform(pos_df)
                        self.preprocess_decisions[f"spacy_pos"][f"pos_pca_{text_col}"] = pca
                        pos_pca_cols = [f'POS PC-1 {text_col}', f'POS PC-2 {text_col}']
                        pos_df = pd.DataFrame(comps, columns=pos_pca_cols)
                        pos_df_pca = pos_df[pos_pca_cols]
                        df = pd.merge(df, pos_df_pca, left_index=True, right_index=True, how='left')
                        nlp_columns.append(text_col)
                    else:
                        pass
                    df.drop(f'nof_words_{text_col}', axis=1, inplace=True)
                except AttributeError:
                    pass
            # get unique pos tag columns
            unique_pos_cols = list(set(pos_columns))
            self.preprocess_decisions[f"spacy_pos"]["pos_tagger_cols"] = unique_pos_cols
            # get unique original column names
            unique_nlp_cols = list(set(nlp_columns))
            self.nlp_columns = unique_nlp_cols
            print(df.info())
        return df

    def pos_tagging_pca(self):
        logging.info('Start spacy POS tagging loop.')
        algorithm = 'spacy_pos'
        if self.prediction_mode:
            text_columns = self.nlp_columns
            self.dataframe = self.dataframe_nlp(
                self.dataframe, text_columns, mode='transform')
            logging.info('Finished spacy POS tagging loop.')
        else:
            X_train, X_test, Y_train, Y_test = self.unpack_test_train_dict()
            self.preprocess_decisions[f"spacy_pos"] = {}
            if not self.nlp_columns:
                text_columns = X_train.select_dtypes(include=['object']).columns
            else:
                text_columns = self.nlp_columns
            X_train = self.dataframe_nlp(X_train,
                                         text_columns, mode='fit'
                                         )
            X_test = self.dataframe_nlp(X_test,
                                        self.nlp_columns, mode='transform'
                                        )
            logging.info('Finished spacy POS tagging loop.')
            return self.wrap_test_train_to_dict(X_train, X_test, Y_train, Y_test)
