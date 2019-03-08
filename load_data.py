from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
from utils import *
import numpy as np


def load_convmfp_data(data_dir, embedding_path, max_vocab_size=20000, item_max_len=1000):
    if not os.path.exists(data_dir):
        sys.exit(get_now_time() + 'invalid directory')
    else:
        print(get_now_time() + 'loading split data')

        # collect all users id and items id
        user_set = set()
        item_set = set()

        def read_from_file(path):
            with open(path, 'r', errors='ignore') as f:
                for line in f.readlines():
                    content = line.strip().split('::')
                    user_set.add(content[0])
                    item_set.add(content[1])

        read_from_file(data_dir + 'train')
        read_from_file(data_dir + 'valid')
        read_from_file(data_dir + 'test')

        # convert id to array index
        index2user = list(user_set)
        index2item = list(item_set)
        user2index = {x: i for i, x in enumerate(index2user)}
        item2index = {x: i for i, x in enumerate(index2item)}

        def read_from_file2(path, max_rating, min_rating):
            tuple_list = []
            with open(path, 'r', errors='ignore') as f:
                for line in f.readlines():
                    content = line.strip().split('::')
                    u = user2index[content[0]]
                    i = item2index[content[1]]
                    r = float(content[2])
                    if max_rating < r:
                        max_rating = r
                    if min_rating > r:
                        min_rating = r
                    tuple_list.append((u, i, r))
            return tuple_list, max_rating, min_rating

        max_r = -1
        min_r = 1e10
        train_tuple_list, max_r, min_r = read_from_file2(data_dir + 'train', max_r, min_r)
        valid_tuple_list, max_r, min_r = read_from_file2(data_dir + 'valid', max_r, min_r)
        test_tuple_list, max_r, min_r = read_from_file2(data_dir + 'test', max_r, min_r)

        user2item_rating, item2user_rating = format_train_data(train_tuple_list, len(user_set), len(item_set))

        doc_list = []
        with open(data_dir + '/train', 'r', errors='ignore') as f:
            for line in f.readlines():
                content = line.strip().split('::')
                review = content[3]
                doc_list.append(review)

        word2index, index2word = get_word2index(doc_list, max_word_num=max_vocab_size)
        embedding = load_word2vec_embedding(embedding_path, word2index)

        item2word_list = {}
        with open(data_dir + '/train', 'r', errors='ignore') as f:
            for line in f.readlines():
                content = line.strip().split('::')
                i = item2index[content[1]]
                review = content[3]

                word_list = []
                for w in review.split(' '):
                    if w in word2index:
                        word_list.append(word2index[w])

                if i in item2word_list:
                    if len(item2word_list[i]) < item_max_len:
                        item2word_list[i].extend(word_list)
                else:
                    item2word_list[i] = word_list

        item_word_list = format_text(item2word_list, item_max_len, word2index['<PAD>'])

    return user2item_rating, item2user_rating, train_tuple_list, valid_tuple_list, test_tuple_list, max_r, min_r, embedding, item_word_list


def format_train_data(train_tuple_list, user_num, item_num):
    user2item2rating = {}
    item2user2rating = {}
    for x in train_tuple_list:
        u = x[0]
        i = x[1]
        r = x[2]
        if u in user2item2rating:
            user2item2rating[u][i] = r
        else:
            user2item2rating[u] = {i: r}
        if i in item2user2rating:
            item2user2rating[i][u] = r
        else:
            item2user2rating[i] = {u: r}

    user2item_rating = []
    for u in range(user_num):
        if u in user2item2rating:
            item2rating = user2item2rating[u]
            index = list(item2rating.keys())
            rating = list(item2rating.values())
            rating = np.asarray(rating, dtype=np.float32)
            user2item_rating.append([index, rating])
        else:
            user2item_rating.append([])

    item2user_rating = []
    for i in range(item_num):
        if i in item2user2rating:
            user2rating = item2user2rating[i]
            index = list(user2rating.keys())
            rating = list(user2rating.values())
            rating = np.asarray(rating, dtype=np.float32)
            item2user_rating.append([index, rating])
        else:
            item2user_rating.append([])

    return user2item_rating, item2user_rating


def format_text(target2word_list, max_word_num, pad_int):
    target_num = len(target2word_list)
    target_word_list = np.full((target_num, max_word_num), pad_int)

    for i in range(target_num):
        word_list = np.asarray(target2word_list[i], dtype=np.int32)
        word_num = len(word_list)
        if word_num > max_word_num:
            target_word_list[i] = word_list[:max_word_num]
        else:
            target_word_list[i][:word_num] = word_list

    return target_word_list


def get_word2index(doc_list, max_word_num=20000):
    def split_words_by_space(text):
        return text.split(' ')

    vectorizer = CountVectorizer(max_features=max_word_num, analyzer=split_words_by_space)
    # descending sorting the dictionary by df
    X = vectorizer.fit_transform(doc_list)
    X[X > 0] = 1
    df = np.ravel(X.sum(axis=0))
    words = vectorizer.get_feature_names()
    word_df = list(zip(words, df))
    word_df.sort(key=lambda x: x[1], reverse=True)
    index2word = [w for (w, f) in word_df]
    index2word.extend(['<UNK>', '<GO>', '<EOS>', '<PAD>'])
    word2index = {w: i for i, w in enumerate(index2word)}

    return word2index, index2word


def load_word2vec_embedding(embedding_path, word2index):
    word2vec = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    existing_embedding = {}
    for w in word2index.keys():
        if w in word2vec:
            existing_embedding[w] = word2vec[w]

    matrix = np.asarray(list(existing_embedding.values()))
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    emb_size = matrix.shape[1]

    embedding = np.zeros((len(word2index), emb_size), dtype=np.float32)
    for (w, idx) in word2index.items():
        if w in existing_embedding:
            embedding[idx] = existing_embedding[w]
        else:
            embedding[idx] = np.random.normal(mean, std)  # sample an embedding from the same distribution if not exist

    return embedding


def get_prediction(tuple_list, U, V):
    new_tuple_list = []
    for x in tuple_list:
        u = x[0]
        i = x[1]
        r = x[2]
        p = U[u].dot(V[i])
        new_tuple_list.append((r, p))

    return new_tuple_list


def load_author_provide_data(data_dir, embedding_path, max_vocab_size=20000, item_max_len=300):
    if not os.path.exists(data_dir):
        sys.exit(get_now_time() + 'invalid directory')
    else:
        print(get_now_time() + 'loading split data')

        # collect all users id and items id
        user_set = set()
        item_set = set()

        def read_from_file2(path, max_rating, min_rating):
            tuple_list = []
            with open(path, 'r', errors='ignore') as f:
                for line in f.readlines():
                    content = line.strip().split('::')
                    u = int(content[0])
                    i = int(content[1])
                    user_set.add(u)
                    item_set.add(i)
                    r = float(content[2])
                    if max_rating < r:
                        max_rating = r
                    if min_rating > r:
                        min_rating = r
                    tuple_list.append((u, i, r))
            return tuple_list, max_rating, min_rating

        max_r = -1
        min_r = 1e10
        train_tuple_list, max_r, min_r = read_from_file2(data_dir + 'train', max_r, min_r)
        valid_tuple_list, max_r, min_r = read_from_file2(data_dir + 'valid', max_r, min_r)
        test_tuple_list, max_r, min_r = read_from_file2(data_dir + 'test', max_r, min_r)

        user2item_rating, item2user_rating = format_train_data(train_tuple_list, len(user_set), len(item_set))

        doc_list = []
        with open(data_dir + 'plot.item', 'r', errors='ignore') as f:
            for line in f.readlines():
                content = line.strip().split('::')
                review = content[1]
                doc_list.append(review)

        word2index, index2word = get_word2index(doc_list, max_word_num=max_vocab_size)
        embedding = load_word2vec_embedding(embedding_path, word2index)

        item2word_list = {}
        with open(data_dir + 'plot.item', 'r', errors='ignore') as f:
            for line in f.readlines():
                content = line.strip().split('::')
                i = int(content[0])
                review = content[1]

                word_list = []
                for w in review.split(' '):
                    if w in word2index:
                        word_list.append(word2index[w])

                item2word_list[i] = word_list

        item_word_list = format_text(item2word_list, item_max_len, word2index['<PAD>'])

    return user2item_rating, item2user_rating, train_tuple_list, valid_tuple_list, test_tuple_list, max_r, min_r, embedding, item_word_list
