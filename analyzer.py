import csv
import os
import string
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx

from nltk import Text, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from pymystem3 import Mystem
from simple_elmo import ElmoModel
from wordcloud import WordCloud
from parser import ConsultantPlusParser
from utils import get_config, generate_file_name_with_postfix
from tqdm import tqdm
from utils.exceptions import ConsultantPlusAnalyzerException
from ipymarkup import show_dep_ascii_markup as show_markup
from razdel import sentenize, tokenize
from navec import Navec
from slovnet import Syntax


class ConsultantPlusAnalyzer:

    def __init__(self, is_elmo_used=False):
        self.config = get_config('config.yml')
        self.parser = ConsultantPlusParser(config=self.config)
        self.model = ElmoModel()
        self.mystem = Mystem()
        self.spec_chars = string.punctuation + '\n\xa0«»\t—…'
        self.stop_words = stopwords.words("russian")
        self.stop_words.extend(
            ['и', 'в', 'на', 'n', 'рф', 'гк', 'юридического', ' ', '1', 'ред',
             '2', 'ст', 'также', 'свой', 'либо', 'это', 'текст', 'закон', 'который', 'иной', 'год', 'мочь', ])
        if is_elmo_used:
            self.model.load(self.config['model_info_file'])
        self.navec = Navec.load(self.config['navec_news_v1_1B_250K_300d_100q'])
        self.syntax = Syntax.load(self.config['slovnet_syntax_news_v1'])
        self.syntax.navec(self.navec)

    def save_information_about_target_words_by_codex_type(self, codex_type, codex_id):
        raw_articles_info = self.parser.sorted_articles_info[codex_type]
        if os.path.exists(generate_file_name_with_postfix(
            self.config['information_about_target_words'],
            str(codex_id))):
            os.remove(generate_file_name_with_postfix(
                self.config['information_about_target_words'],
                str(codex_id)))
        with open(generate_file_name_with_postfix(
            self.config['information_about_target_words'],
            str(codex_id)),
            mode='w') as information_about_target_words_file:
            information_about_target_words_writer = csv.writer(
                information_about_target_words_file, delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            information_about_target_words_writer.writerow(
                ['article_id', 'article_title', 'parts_after_target_words', 'sentences'])
            for article_info in tqdm(raw_articles_info):
                text = self.parser.get_article_text_by_id(article_info.id)
                if text.find('если иное не предусмотрено') != -1:
                    text_parts = text.split('если иное не предусмотрено')
                    parts_before_target_words = list()
                    for i in range(0, len(text_parts) - 1):
                        parts_before_target_words.append(text_parts[i].split('.')[-1])
                    parts_after_target_words = list()
                    for i in range(1, len(text_parts)):
                        parts_after_target_words.append(text_parts[i].split('.')[0])
                    sentences = list()
                    for i in range(len(parts_before_target_words)):
                        sentences.append(parts_before_target_words[i] + 'если иное не предусмотрено' + parts_after_target_words[i])
                    information_about_target_words_writer.writerow([article_info.id, article_info.title, '~'.join(parts_after_target_words), '~'.join(sentences)])

    def plot_word_vectors_graph(self, proximity_threshold, count_of_words=None):
        # TODO: найти порог близости для каждой статьи
        articles_vectors_info = dict()
        articles_words_info = dict()
        with open(self.config[
                      'articles_vectors_info_file']) as articles_vectors_info_file_with_words:
            reader = csv.reader(articles_vectors_info_file_with_words)
            for row in tqdm(reader):
                article_id = int(row[0])
                vector = self.convert_vector_from_string_value(
                    row[1:len(row) - 1])
                word = row[-1]
                if article_id not in articles_vectors_info:
                    articles_vectors_info[article_id] = list()
                    articles_words_info[article_id] = list()
                articles_vectors_info[article_id].append(vector)
                articles_words_info[article_id].append(word)
        for article_id, article_vectors in tqdm(articles_vectors_info.items()):
            n = len(article_vectors) if not count_of_words else count_of_words
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i][j] = self.get_euclidean_distance(
                        article_vectors[i], article_vectors[j])
            G = nx.Graph()
            edges_info = dict()
            for i in range(n):
                for j in range(n):
                    if dist_matrix[i][j] > proximity_threshold and ((
                                                                    articles_words_info[
                                                                        article_id][
                                                                        i],
                                                                    articles_words_info[
                                                                        article_id][
                                                                        j]) not in edges_info or (
                                                                    articles_words_info[
                                                                        article_id][
                                                                        j],
                                                                    articles_words_info[
                                                                        article_id][
                                                                        i]) not in edges_info):
                        edges_info[(articles_words_info[article_id][i],
                                    articles_words_info[article_id][
                                        j])] = round(dist_matrix[i][j], 2)
            G.add_weighted_edges_from(
                [(item[0][0], item[0][1], item[1]) for item in
                 edges_info.items()])
            pos = nx.spring_layout(G)
            plt.figure(figsize=(50, 50))
            nx.draw(G, pos, node_size=10000, with_labels=True)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edges_info)
            plt.show()

    def save_syntax_analysis_by_text(self, text, file, is_many_sentences=False):
        f = open(file, 'a')
        sys.stdout = f
        print('-' * 100)
        if text != 'None':
            if not is_many_sentences:
                chunk = list()
                for sent in sentenize(text):
                    tokens = [_.text for _ in tokenize(sent.text)]
                    chunk.append(tokens)
                markup = next(self.syntax.map(chunk))
                words, deps = list(), list()
                for token in markup.tokens:
                    words.append(token.text)
                    source = int(token.head_id) - 1
                    target = int(token.id) - 1
                    if source > 0 and source != target:
                        deps.append([source, target, token.rel])
                show_markup(words, deps)
            else:
                for sentence in text.split('.'):
                    if len(sentence.split()) > 5:
                        chunk = list()
                        for sent in sentenize(sentence):
                            tokens = [_.text for _ in tokenize(sent.text)]
                            chunk.append(tokens)
                        markup = next(self.syntax.map(chunk))
                        words, deps = list(), list()
                        for token in markup.tokens:
                            words.append(token.text)
                            source = int(token.head_id) - 1
                            target = int(token.id) - 1
                            if source > 0 and source != target:
                                deps.append([source, target, token.rel])
                        show_markup(words, deps)
        else:
            print('None')
        print('-' * 100)

    def get_words_matrix_variance(self):
        articles_vectors_info = dict()
        articles_words_info = dict()
        with open(self.config[
                      'articles_vectors_info_file']) as articles_vectors_info_file_with_words:
            reader = csv.reader(articles_vectors_info_file_with_words)
            for row in tqdm(reader):
                article_id = int(row[0])
                vector = self.convert_vector_from_string_value(
                    row[1:len(row) - 1])
                word = row[-1]
                if article_id not in articles_vectors_info:
                    articles_vectors_info[article_id] = list()
                    articles_words_info[article_id] = list()
                articles_vectors_info[article_id].append(vector)
                articles_words_info[article_id].append(word)
        for article_id, article_vectors in tqdm(articles_vectors_info.items()):
            mat = np.array(article_vectors)
            print(
                f'The variance in article {article_id} is {mat.var()}')

    def get_prediction(self, words=None, file_with_vectors=None):
        words_vectors = list()
        if file_with_vectors:
            with open(file_with_vectors) as file:
                reader = csv.reader(file)
                for row in reader:
                    words_vectors.append(
                        self.convert_vector_from_string_value(row))
        else:
            for word in tqdm(words):
                words_vectors.append(self.model.get_elmo_vectors(word)[0][0])
        articles_vectors_info = dict()
        articles_words_info = dict()
        with open(self.config[
                      'articles_vectors_info_file']) as articles_vectors_info_file_with_words:
            reader = csv.reader(articles_vectors_info_file_with_words)
            for row in tqdm(reader):
                article_id = int(row[0])
                vector = self.convert_vector_from_string_value(
                    row[1:len(row) - 1])
                word = row[-1]
                if article_id not in articles_vectors_info:
                    articles_vectors_info[article_id] = list()
                    articles_words_info[article_id] = list()
                articles_vectors_info[article_id].append(vector)
                articles_words_info[article_id].append(word)
        articles_distance_info = list()
        for word_vector in tqdm(words_vectors):
            articles_distance_info.append(dict())
            for article_id in tqdm(articles_vectors_info):
                if article_id not in articles_distance_info[-1]:
                    articles_distance_info[-1][article_id] = list()
                for vector in tqdm(articles_vectors_info[article_id]):
                    articles_distance_info[-1][article_id].append(
                        self.get_euclidean_distance(word_vector, vector))
        articles_average_distance_info = list()
        for info in tqdm(articles_distance_info):
            articles_average_distance_info.append(dict())
            for article_id in tqdm(info):
                articles_average_distance_info[-1][article_id] = np.average(
                    np.array(info[article_id]))
        prediction_articles_id = list()
        for info in articles_average_distance_info:
            id = -1
            min_dist = sys.maxsize
            for article_id, dist in info.items():
                if dist < min_dist:
                    id = article_id
                    min_dist = dist
            prediction_articles_id.append(id)
        print(prediction_articles_id)

    def save_unique_words_in_articles_analysis(self, codex_type, codex_id):
        raw_articles_info = self.parser.sorted_articles_info[codex_type]
        articles_info = list()
        for article_info in tqdm(raw_articles_info):
            text = self.parser.get_article_text_by_id(article_info.id)
            text = text.lower()
            text = self.remove_chars_from_text(text, self.spec_chars)
            article_tokens = word_tokenize(
                ' '.join(self.mystem.lemmatize(text)))
            for stop_word in self.stop_words:
                while stop_word in article_tokens:
                    article_tokens.remove(stop_word)
            text = Text(article_tokens)
            f_dist = FreqDist(text)
            f_dist = list(filter(lambda item: item[1] == 1, f_dist.items()))
            articles_info.append(
                (article_info.id, len(f_dist) / len(article_tokens)))
        if os.path.exists(generate_file_name_with_postfix(
            self.config['unique_words_in_articles_analysis_file'],
            str(codex_id))):
            os.remove(generate_file_name_with_postfix(
                self.config['unique_words_in_articles_analysis_file'],
                str(codex_id)))
        with open(generate_file_name_with_postfix(
            self.config['unique_words_in_articles_analysis_file'],
            str(codex_id)),
            mode='w') as unique_words_in_articles_analysis_file:
            unique_words_in_articles_analysis_writer = csv.writer(
                unique_words_in_articles_analysis_file, delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            unique_words_in_articles_analysis_writer.writerow(
                ['article_id', 'unique_words_frequency'])
            for frequency_info in articles_info:
                unique_words_in_articles_analysis_writer.writerow(
                    [frequency_info[0], frequency_info[1]])

    def save_most_popular_words_analysis(self, most_common_quantity):
        articles_tokens = list()
        for (codex_type, _) in tqdm(self.parser.codex_urls):
            raw_articles_info = self.parser.sorted_articles_info[codex_type]
            for article_info in tqdm(raw_articles_info):
                text = self.parser.get_article_text_by_id(article_info.id)
                text = text.lower()
                text = self.remove_chars_from_text(text, self.spec_chars)
                article_tokens = word_tokenize(
                    ' '.join(self.mystem.lemmatize(text)))
                for stop_word in self.stop_words:
                    while stop_word in article_tokens:
                        article_tokens.remove(stop_word)
                articles_tokens.extend(article_tokens)
        text = Text(articles_tokens)
        f_dist = FreqDist(text)
        if os.path.exists(self.config['most_popular_words_analysis_file']):
            os.remove(self.config['most_popular_words_analysis_file'])
        with open(self.config['most_popular_words_analysis_file'],
                  mode='w') as most_popular_words_analysis_file:
            most_popular_words_analysis_writer = csv.writer(
                most_popular_words_analysis_file, delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            most_popular_words_analysis_writer.writerow(
                ['word', 'word_count', 'frequency'])
            for info in f_dist.most_common(most_common_quantity):
                most_popular_words_analysis_writer.writerow(
                    [info[0], info[1], info[1] / len(articles_tokens)])

    def save_unique_words_analysis(self, uniqueness_threshold):
        """Сохраняем информацию о количестве уникальных слов и количестве статей, в которых эти слова встречаются, а также информацию о заданном количестве уникальных слов"""
        articles_tokens = list()
        articles_words_info = dict()
        for (codex_type, _) in tqdm(self.parser.codex_urls):
            raw_articles_info = self.parser.sorted_articles_info[codex_type]
            for article_info in tqdm(raw_articles_info):
                text = self.parser.get_article_text_by_id(article_info.id)
                text = text.lower()
                text = self.remove_chars_from_text(text, self.spec_chars)
                article_tokens = word_tokenize(
                    ' '.join(self.mystem.lemmatize(text)))
                for stop_word in self.stop_words:
                    while stop_word in article_tokens:
                        article_tokens.remove(stop_word)
                articles_words_info[
                    self.get_unique_article_identifier(codex_type,
                                                       article_info.id)] = list(
                    set(article_tokens))
                articles_tokens.extend(article_tokens)
        text = Text(articles_tokens)
        f_dist = FreqDist(text)
        f_dist = list(filter(lambda item: item[1] <= uniqueness_threshold,
                             f_dist.items()))
        unique_words_info = dict()
        # Сохраняем информацию в виде: 'уникальное слово': ['количество во всем корпусе', 'количество статей, в котром встретилось это слово']
        for word_info in f_dist:
            if word_info[0] not in unique_words_info:
                unique_words_info[word_info[0]] = [word_info[1], 0]
            for article_id in tqdm(articles_words_info):
                if word_info[0] in articles_words_info[article_id]:
                    unique_words_info[word_info[0]][1] += 1
        if os.path.exists(self.config['articles_unique_words_info_file']):
            os.remove(self.config['articles_unique_words_info_file'])
        with open(self.config['articles_unique_words_info_file'],
                  mode='w') as articles_unique_words_info_file:
            articles_unique_words_info_writer = csv.writer(
                articles_unique_words_info_file, delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            articles_unique_words_info_writer.writerow(
                ['word', 'word_count', 'articles_count'])
            for info in unique_words_info.items():
                articles_unique_words_info_writer.writerow(
                    [info[0], info[1][0], info[1][1]])
        unique_words_metrics = dict()
        # Сохраняем информацию в виде: 'заданное количество слова во всем корпусе': 'количество таких слов во всем корпусе'
        for value in unique_words_info.values():
            if value[0] not in unique_words_metrics:
                unique_words_metrics[value[0]] = value[1]
            else:
                unique_words_metrics[value[0]] += value[1]
        if os.path.exists(self.config['articles_unique_words_analysis_file']):
            os.remove(self.config['articles_unique_words_analysis_file'])
        with open(self.config['articles_unique_words_analysis_file'], mode='w') as articles_unique_words_analysis_file:
            articles_unique_words_analysis_writer = csv.writer(articles_unique_words_analysis_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            articles_unique_words_analysis_writer.writerow(['count_unique_words_frequency', 'count_unique_words_in_corpus'])
            for info in unique_words_metrics.items():
                articles_unique_words_analysis_writer.writerow([info[0], info[1]])
        if os.path.exists(self.config['articles_unique_words_analysis_file_with_frequency']):
            os.remove(self.config['articles_unique_words_analysis_file_with_frequency'])
        with open(self.config['articles_unique_words_analysis_file_with_frequency'], mode='w') as articles_unique_words_analysis_file:
            articles_unique_words_analysis_writer = csv.writer(articles_unique_words_analysis_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            articles_unique_words_analysis_writer.writerow(['count_unique_words_frequency', 'count_unique_words_in_corpus_frequency'])
            for info in unique_words_metrics.items():
                articles_unique_words_analysis_writer.writerow([info[0], info[1] / len(articles_tokens)])

    def save_codex_hist_info(self, codex_type, codex_id, constraint=None):
        """Сохранение частотности слов во всем корпусе"""
        raw_articles_info = self.parser.sorted_articles_info[codex_type]
        articles_tokens = list()
        for article_info in tqdm(raw_articles_info):
            text = self.parser.get_article_text_by_id(article_info.id)
            text = text.lower()
            text = self.remove_chars_from_text(text, self.spec_chars)
            article_tokens = word_tokenize(
                ' '.join(self.mystem.lemmatize(text)))
            for stop_word in self.stop_words:
                while stop_word in article_tokens:
                    article_tokens.remove(stop_word)
            articles_tokens.extend(article_tokens)
        text = Text(articles_tokens)
        f_dist = FreqDist(text)
        if not constraint:
            if os.path.exists(generate_file_name_with_postfix(
                self.config['articles_frequency_info_file'], str(codex_id))):
                os.remove(generate_file_name_with_postfix(
                    self.config['articles_frequency_info_file'], str(codex_id)))
            with open(generate_file_name_with_postfix(
                self.config['articles_frequency_info_file'], str(codex_id)),
                      mode='w') as articles_frequency_info_file:
                articles_frequency_info_writer = csv.writer(
                    articles_frequency_info_file, delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)
                articles_frequency_info_writer.writerow(['word', 'frequency'])
                for frequency_info in f_dist.most_common(100):
                    articles_frequency_info_writer.writerow([frequency_info[0],
                                                             frequency_info[
                                                                 1] / len(
                                                                 articles_tokens)])
        else:
            if os.path.exists(generate_file_name_with_postfix(
                self.config['articles_frequency_info_file_with_constraint'],
                str(codex_id))):
                os.remove(generate_file_name_with_postfix(
                    self.config['articles_frequency_info_file_with_constraint'],
                    str(codex_id)))
            with open(generate_file_name_with_postfix(
                self.config['articles_frequency_info_file_with_constraint'],
                str(codex_id)),
                mode='w') as articles_frequency_info_file:
                articles_frequency_info_writer = csv.writer(
                    articles_frequency_info_file, delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)
                articles_frequency_info_writer.writerow(['word', 'frequency'])
                f_dist = list(
                    filter(lambda item: item[1] > constraint, f_dist.items()))
                for frequency_info in f_dist:
                    articles_frequency_info_writer.writerow([frequency_info[0],
                                                             frequency_info[
                                                                 1] / len(
                                                                 articles_tokens)])

    def save_word_vectors_analysis_info(self, codex_type, most_common_count):
        articles_info = dict()
        raw_articles_info = self.parser.sorted_articles_info[codex_type]
        i = 0
        for article_info in tqdm(raw_articles_info):
            text = self.parser.get_article_text_by_id(article_info.id)
            text = text.lower()
            text = self.remove_chars_from_text(text, self.spec_chars)
            article_tokens = word_tokenize(
                ' '.join(self.mystem.lemmatize(text)))
            for stop_word in self.stop_words:
                while stop_word in article_tokens:
                    article_tokens.remove(stop_word)
            article_vectors = list()
            article_words = list()
            text = Text(article_tokens)
            f_dist = FreqDist(text)
            for token in tqdm(f_dist.most_common(most_common_count)):
                vector = self.model.get_elmo_vectors(token[0])
                article_words.append(token[0])
                article_vectors.append(vector[0][0])
            articles_info[article_info.id] = [article_vectors, article_words]
            i += 1
            if i == 20:
                break
        if os.path.exists(self.config['articles_vectors_info_file']):
            os.remove(self.config['articles_vectors_info_file'])
        with open(self.config['articles_vectors_info_file'],
                  mode='w') as articles_vectors_info_file:
            articles_vectors_info_writer = csv.writer(
                articles_vectors_info_file, delimiter=',',
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL)
            for article_id, article_vectors_info in articles_info.items():
                for i, article_vector_info in enumerate(
                    article_vectors_info[0]):
                    articles_vectors_info_writer.writerow(
                        [article_id, *article_vector_info,
                         article_vectors_info[1][i]])

    def frequency_analysis_of_words(self):
        articles_tokens = list()
        for i in tqdm(range(10)):
            text = self.parser.get_article_text_by_id(i)
            text = text.lower()
            text = self.remove_chars_from_text(text, self.spec_chars)
            articles_tokens.extend(
                word_tokenize(' '.join(self.mystem.lemmatize(text))))
        for stop_word in self.stop_words:
            while stop_word in articles_tokens:
                articles_tokens.remove(stop_word)
        raw_text = ' '.join(articles_tokens)
        word_cloud = WordCloud().generate(raw_text)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        text = Text(articles_tokens)
        f_dist = FreqDist(text)
        f_dist.plot(30, cumulative=False)

    def links_on_target_words_analysis(self):
        for codex_id in tqdm(range(len(self.parser.codex_urls))):
            if os.path.exists(generate_file_name_with_postfix(
                self.config['information_about_target_words_with_links'],
                str(codex_id))):
                os.remove(generate_file_name_with_postfix(
                    self.config['information_about_target_words_with_links'],
                    str(codex_id)))
            with open(generate_file_name_with_postfix(
                self.config['information_about_target_words_with_links'],
                str(codex_id)),
                mode='w') as information_about_target_words_file:
                information_about_target_words_writer = csv.writer(
                    information_about_target_words_file, delimiter=',',
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL)
                information_about_target_words_writer.writerow(
                    ['article_id', 'article_title', 'article_url', 'links_on_target'])
                target_words_info = pd.read_csv(generate_file_name_with_postfix(self.config['information_about_target_words'], str(codex_id)))
                for row in tqdm(target_words_info.itertuples()):
                    links_on_target = list()
                    for part_of_target_words in row[3].split('~'):
                        if self.parser.get_links_on_target_words_by_id_and_target_words(row[1], part_of_target_words):
                            links_on_target.append(self.parser.get_links_on_target_words_by_id_and_target_words(row[1], part_of_target_words))
                        else:
                            links_on_target.append('None')
                    information_about_target_words_writer.writerow([row[1], row[2], self.parser.get_article_url_by_id(row[1]), ' '.join(links_on_target)])

    @staticmethod
    def save_syntax_analysis(analyzer):
        for codex_id in tqdm(range(len(analyzer.parser.codex_urls))):
            target_words_info = pd.read_csv(generate_file_name_with_postfix(
                analyzer.config['information_about_target_words'], str(codex_id)))
            for row in tqdm(target_words_info.itertuples()):
                for sentence in row[-1].split('~'):
                    analyzer.save_syntax_analysis_by_text(sentence, generate_file_name_with_postfix(analyzer.config['article_target_words_realation_info'], str(row[1])))

    @staticmethod
    def save_syntax_analysis_in_links(analyzer):
        for codex_id in tqdm(range(len(analyzer.parser.codex_urls))):
            target_words_info = pd.read_csv(generate_file_name_with_postfix(analyzer.config['information_about_target_words_with_links'], str(codex_id)))
            for row in tqdm(target_words_info.itertuples()):
                for url in row[-1].split(' '):
                    if url != 'None':
                        analyzer.save_syntax_analysis_by_text(analyzer.parser.get_text_by_url(url), generate_file_name_with_postfix(analyzer.config['article_target_words_in_links_realation_info'], str(row[1])), is_many_sentences=True)
                    else:
                        analyzer.save_syntax_analysis_by_text('None', generate_file_name_with_postfix(analyzer.config['article_target_words_in_links_realation_info'], str(row[1])))

    @staticmethod
    def plot_frequency_analysis_of_words(analyzer, is_constraint=None):
        """Построение частотности слов во всем корпусе"""
        if not is_constraint:
            for i in range(10):
                data = pd.read_csv(generate_file_name_with_postfix(
                    analyzer.config['articles_frequency_info_file'], str(i)),
                    delimiter=',')
                data.plot(x='word', y='frequency', figsize=(50, 7),
                          kind='scatter')
                plt.xticks(rotation=60)
                plt.show()
        else:
            for i in range(10):
                data = pd.read_csv(generate_file_name_with_postfix(
                    analyzer.config[
                        'articles_frequency_info_file_with_constraint'],
                    str(i)),
                    delimiter=',')
                data = data.sort_values(by='frequency', axis='index')
                data.plot(x='word', y='frequency', figsize=(50, 7),
                          kind='scatter')
                plt.xticks(rotation=60)
                plt.show()

    @staticmethod
    def plot_unique_words_in_articles_analysis(analyzer):
        """Графики частотности уникальных слов в каждом кодексе по article_id"""
        for i in range(10):
            data = pd.read_csv(generate_file_name_with_postfix(
                analyzer.config['unique_words_in_articles_analysis_file'],
                str(i)),
                delimiter=',')
            data = data.sort_values('unique_words_frequency')
            data.plot(x='article_id', y='unique_words_frequency',
                      kind='scatter')
            plt.show()

    @staticmethod
    def plot_unique_words_in_articles_analysis_on_one_graph(analyzer):
        """График частотности уникальных слов в каждом кодексе на одном графике с отсортированной частотностью"""
        data = pd.read_csv(generate_file_name_with_postfix(
                analyzer.config['unique_words_in_articles_analysis_file'],
                str(0)),
                           delimiter=',')
        for i in range(1, 10):
            data = pd.concat([data, pd.read_csv(generate_file_name_with_postfix(
                analyzer.config['unique_words_in_articles_analysis_file'],
                str(i)),
                delimiter=',')])
        data['article_id'] = data.apply(lambda row: row['article_id'] / data['article_id'].max(), axis=1)
        data = data.sort_values('unique_words_frequency')
        data = data.reset_index()
        data.drop('article_id', axis='columns', inplace=True)
        data.drop('index', axis='columns', inplace=True)
        data.plot()
        plt.show()

    @staticmethod
    def plot_unique_words_analysis(analyzer, is_frequency_analysis=False):
        """Построение графика анализа уникальных слов"""
        if not is_frequency_analysis:
            data = pd.read_csv(
                analyzer.config['articles_unique_words_analysis_file'])
        else:
            data = pd.read_csv(
                analyzer.config[
                    'articles_unique_words_analysis_file_with_frequency'])
        data.plot(x='count_unique_words_frequency', y='count_unique_words_in_corpus', kind='scatter')
        plt.show()
        plt.hist(data.count_unique_words_frequency, weights=data.count_unique_words_in_corpus)
        plt.show()

    @staticmethod
    def plot_most_popular_words_analysis(analyzer):
        """Построение графика частотности самых популярных во всем корпусе слов"""
        data = pd.read_csv(analyzer.config['most_popular_words_analysis_file'])
        plt.hist(data.word_count, weights=data.frequency)
        plt.show()
        data.plot(x='word', y='frequency', kind='scatter', figsize=(50, 7))
        plt.xticks(rotation=60)
        plt.show()

    @staticmethod
    def remove_chars_from_text(text, chars):
        return ''.join([ch for ch in text if ch not in chars])

    @staticmethod
    def convert_vector_from_string_value(vector):
        return list(map(lambda value: float(value), vector))

    @staticmethod
    def get_euclidean_distance(vector1, vector2):
        if len(vector1) != len(vector2):
            raise ConsultantPlusAnalyzerException(
                'It is not possible to compare vectors of different dimensions')
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        return np.linalg.norm(v1 - v2)

    @staticmethod
    def get_unique_article_identifier(codex_type, article_id):
        return codex_type + '_' + str(article_id)


if __name__ == '__main__':
    analyzer = ConsultantPlusAnalyzer(is_elmo_used=False)
    # analyzer.plot_frequency_analysis_of_words(analyzer)
    # print(analyzer.parser.get_text_by_url('/document/cons_doc_LAW_368629/41b49109d1875673ebbf6cef045eff7f1afe2d19/#dst27'))
    # analyzer.save_syntax_analysis_in_links(analyzer)
    # analyzer.save_syntax_analysis_by_text('Товарищество собственников жилья создается без ограничения срока деятельности, если иное не предусмотрено уставом товарищества', 'resources/lol.txt')
    # analyzer.links_on_target_words_analysis()
    # print(analyzer.parser.get_sentence_with_target_words_by_id_and_target_words(1, 'настоящим Кодексом'))
    # for i, (codex_type, _) in tqdm(enumerate(analyzer.parser.codex_urls)):
    #     analyzer.save_information_about_target_words_by_codex_type(codex_type, i)
    # analyzer.plot_unique_words_in_articles_analysis_on_one_graph(analyzer)
    # analyzer.get_words_matrix_variance()
    # analyzer.plot_most_popular_words_analysis()
    # analyzer.get_prediction(file_with_vectors='./resources/test_words.csv')
    # analyzer.plot_most_popular_words_analysis()
    # analyzer.plot_unique_words_analysis(is_frequency_analysis=True)
    # analyzer.get_words_matrix_variance()
    analyzer.plot_word_vectors_graph(9, 5)
    # for (codex_type, _) in tqdm(analyzer.parser.codex_urls):
    #     analyzer.save_word_vectors_analysis_info(codex_type, 25)
    #     break
