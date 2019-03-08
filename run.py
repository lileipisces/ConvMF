from load_data import load_convmfp_data, get_prediction
from cnn_module import CNN
from utils import *
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-gd', '--gpu_device', type=str, help='device(s) on GPU, default=0', default='0')
parser.add_argument('-rp', '--raw_data_path', type=str, help='path for loading raw data', default=None)
parser.add_argument('-rt', '--data_ratio', type=str, help='ratio of train:valid:test', default='8:1:1')
parser.add_argument('-pd', '--processed_data_dir', type=str, help='the directory for saving processed data if -rp not None. load processed data from this directory if -rp None', default=None)
parser.add_argument('-ep', '--embedding_path', type=str, help='the path for loading pre-trained embedding', default=None)

parser.add_argument('-ic', '--max_word_count_for_item', type=int, help='If the aggregated text for an item is longer than -ic, only first -ic words are kept, default=1000', default=1000)
parser.add_argument('-mw', '--max_word_num', type=int, help='max word num, default=20000', default=20000)
parser.add_argument('-bs', '--batch_size', type=int, help='batch size, default=128', default=128)
parser.add_argument('-it', '--max_iter', type=int, help='max iteration number, default=50', default=50)
parser.add_argument('-lr', '--learning_rate', type=float, help='learning rate, default=0.001', default=0.001)
parser.add_argument('-dk', '--dropout_keep', type=float, help='dropout ratio, default=0.8', default=0.8)
parser.add_argument('-lu', '--lambda_u', type=float, help='trade-off parameter on user latent factor, default=100', default=100)
parser.add_argument('-lv', '--lambda_v', type=float, help='trade-off parameter on item latent factor, default=10', default=10)
parser.add_argument('-dm', '--dimension', type=int, help='dimension of latent factor, default=20', default=20)
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
if args.raw_data_path is not None and args.processed_data_dir is None:
    sys.exit(get_now_time() + 'provide processed_data_dir for saving data')
elif args.raw_data_path is None and args.processed_data_dir is None:
    sys.exit(get_now_time() + 'provide processed_data_dir for loading data')
if args.embedding_path is None:
    sys.exit(get_now_time() + 'provide embedding_path for loading data')


if args.raw_data_path is not None:
    ratio = [float(r) for r in args.data_ratio.split(':')]
    split_raw_data(args.raw_data_path, args.processed_data_dir, ratio[0], ratio[1], ratio[2])
user2item_rating, item2user_rating, train_tuple_list, valid_tuple_list, test_tuple_list, max_r, min_r, embedding, \
item_word_list = load_convmfp_data(args.processed_data_dir, args.embedding_path, args.max_word_num, args.max_word_count_for_item)


U = np.random.uniform(size=(len(user2item_rating), args.dimension))  # (user_num, dimension)
V = np.random.uniform(size=(len(item2user_rating), args.dimension))  # (item_num, dimension)
I_K = np.eye(args.dimension)  # (dimension, dimension)
item_cnn = CNN(item_word_list, args.max_word_count_for_item, embedding, args.dimension, args.batch_size, args.learning_rate, args.dropout_keep)
V_ = item_cnn.get_latent_factor()

# early stop setting
pre_val_eval = 1e10
for it in range(1, args.max_iter + 1):
    print(get_now_time() + 'iteration {}'.format(it))

    for u in range(len(user2item_rating)):
        item_rating = user2item_rating[u]
        if len(item_rating) != 0:
            index = item_rating[0]
            R_i = item_rating[1]
            V_temp = V[index]

            matrix_A = V_temp.T.dot(V_temp) + args.lambda_u * I_K  # (dimension, dimension)
            matrix_b = V_temp.T.dot(R_i)  # (dimension,)
            U[u] = np.linalg.solve(matrix_A, matrix_b)
        else:
            U[u] = np.zeros((args.dimension,))
    print(get_now_time() + 'updated U')

    for i in range(len(item2user_rating)):
        user_rating = item2user_rating[i]
        if len(user_rating) != 0:
            index = user_rating[0]
            R_u = user_rating[1]
            U_temp = U[index]

            matrix_A = U_temp.T.dot(U_temp) + args.lambda_v * I_K  # (dimension, dimension)
            matrix_b = U_temp.T.dot(R_u) + args.lambda_v * V_[i]  # (dimension,)
            V[i] = np.linalg.solve(matrix_A, matrix_b)
        else:
            V[i] = V_[i]
    print(get_now_time() + 'updated V')

    # evaluating
    train_predict = get_prediction(train_tuple_list, U, V)
    train_rmse = root_mean_square_error(train_predict, max_r, min_r)
    print(get_now_time() + 'RMSE on train data: {}'.format(train_rmse))
    valid_predict = get_prediction(valid_tuple_list, U, V)
    valid_rmse = root_mean_square_error(valid_predict, max_r, min_r)
    print(get_now_time() + 'RMSE on valid data: {}'.format(valid_rmse))
    test_predict = get_prediction(test_tuple_list, U, V)
    test_rmse = root_mean_square_error(test_predict, max_r, min_r)
    print(get_now_time() + 'RMSE on test data: {}'.format(test_rmse))

    # early stop setting
    if valid_rmse > pre_val_eval:
        print(get_now_time() + 'early stopped')
        break
    pre_val_eval = valid_rmse

    item_cnn.train_one_epoch(V)
    V_ = item_cnn.get_latent_factor()
    print(get_now_time() + 'updated CNN')
