import abc
import spacy #load spacy
import re
import nltk
from gensim.summarization.summarizer import summarize


class Preprocessing(metaclass=abc.ABCMeta):

    """This class provides all preprocessing functions."""

    def __init__(self):

        nltk.download('stopwords')

        from nltk.corpus import stopwords

        # Creating a Spacy english language parser
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'tagger', 'ner'])

        # collecting all english language stop words from nltk
        self.stops = stopwords.words("english")

    """
    This function is used to removes stopwords, puncuations, and will lemmatize the input text 

    Parameters
        ----------
        comment : str,
            The string you want to normalize
        lowercase : boolean, (default True)
            If you want to convert text to lower case, this may not be always (ex: for entity recognition)

    Return:
        ----------
        clean normalized text
    """
    def normalize(self, comment, lowercase=True, lemmatize=True, only_alpha=True):
        count = True
        if lowercase:
            comment = comment.lower()

        if only_alpha:
            comment = re.sub(r'([^a-zA-Z\s]+?)', ' ', comment)
        comment = self.nlp(comment)

        lemmatized = list()

        for word in comment:
            if not (word.is_punct or word.is_digit):
                if lemmatize:
                    lemma = word.lemma_.strip()
                else:
                    lemma = word

                if lemma and count:
                    if lemma not in self.stops:
                        if re.compile(r"\(|\[|\<").search(lemma) is not None:
                            if re.compile(r"\(|\[|\<").search(lemma).span()[0] == 0:
                                count = False
                            else:
                                lemmatized.append(lemma.split('(')[0])
                                count = False
                        else:
                            lemmatized.append(lemma)

                if (not count) and (re.compile(r"\)|\]|\>").search(lemma) is not None):
                    count = True

        return " ".join(lemmatized).replace('\ufeff1', '1')

    """
    This function is used to summarize input using gensim summerizer (Extractive text summarization)

    Parameters
        ----------
        detailed_text : str,
            The Text you want to Summarize (this text should contain how many words it have to summarize
            and has to be appended to the text with "$word_count_split$" in between text and count of words.)

            It is implemented in this way so, it takes advantage pandas apply function (much faster than for loop)

    Return:
        ----------
        summarized text

    Raises
        ------
        exception:
            It raises exception if input text has only one sentence,
            here this is handled by retuning the input text as it is.

    """
    @staticmethod
    def summarize_input(detailed_text, word_limit):

        text = detailed_text.split('$word_count_split$')[0].strip()
        count = int(detailed_text.split('$word_count_split$')[1])

        try:
            summarized_text = ''.join(summarize(text=text, word_count=word_limit - count, split=True))
        except Exception as e:
            print(e, detailed_text)
            summarized_text = text

        return summarized_text

    """
    This function acts as wrapper for text preproseccing (calling all other functions)
    Parameters
        ----------
        target_df : pd.DataFrame,

            pass the dataframe which you want to preprocess
    Return:
        ----------
        preprocessed df

    """
    def pre_process_text(self, target_df, columns,  lowercase=True, lemmatize=True, only_alpha=True):

        target_df = target_df[columns]
        target_df.dropna(inplace=True)

        target_df = target_df.reset_index()

        for i in columns:
            target_df[i] = target_df[i].apply(self.normalize, lowercase=lowercase, lemmatize=lemmatize, only_alpha=only_alpha)

        return target_df

    """
    This function acts as wrapper for text summarization 
    Parameters
        ----------
        target_df : pd.DataFrame,

            pass the dataframe which you want to preprocess
        column : 
            pass the column name which you want to summarize
    Return:
        ----------
        preprocessed df
    """
    def summarize_text(self, target_df, column_dict, word_limit):

        count =0
        for i in column_dict.keys():
            if not column_dict[i]:
                count += target_df[i].str.split().str.len()

        print(count)

        for j in column_dict.keys():
            if column_dict[j]:
                temp_cleaned_detailed_text = target_df[j] + ' $word_count_split$ ' + count.astype(str)
                target_df[j] = temp_cleaned_detailed_text.apply(self.summarize_input, word_limit = word_limit)

        return target_df
