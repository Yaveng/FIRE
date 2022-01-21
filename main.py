import os
import time
import torch
from warnings import simplefilter
from dataloader import Dataset
from model import FIRE
import metric
import warnings
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='Set the hyper-parameters for FIRE')
    parser.add_argument('--dataset', type = str, default='ml1m', help='The name of dataset.')
    parser.add_argument('--pos_type', type = str, default='[4.0, 5.0]', help='The type of positive interactions')
    parser.add_argument('--topks', type= str, default='[5,10]')
    parser.add_argument('--num_his_month', type=int, default=6)
    parser.add_argument('--num_cur_month', type=int, default=1)

    parser.add_argument('--decay_factor', type=float, default=1e-6, help='')
    parser.add_argument('--pri_factor', type=int, default=128, help='')
    parser.add_argument('--alphas', type=str, default='[0.2, 0.2, 0.2, 0.2]', help='')
    parser.add_argument('--use_user_si', action='store_true')
    parser.add_argument('--use_item_si', action='store_true')
    parser.add_argument('--user_threshold', type=float, default=0.6, help='')
    parser.add_argument('--item_threshold', type=float, default=0.55 , help='')

    args = parser.parse_args()
    print(args)

    dataloader = Dataset(args.dataset, num_his_month=args.num_his_month, num_cur_month=args.num_cur_month)
    init_adj_mat, init_time_mat = dataloader.get_init_mats()
    adj_mat, time_mat, cur_time, user_interacted_items_dict, test_user_items_dict = dataloader.get_train_test_data()
    user_sim_mat, item_sim_mat = dataloader.get_user_item_sim_mat(args.use_user_si, args.user_threshold, args.use_item_si, args.item_threshold)
    lm = FIRE(init_adj_mat, init_time_mat, decay_factor=args.decay_factor, pri_factor=args.pri_factor, alpha_list=eval(args.alphas),
              use_user_si=args.use_user_si, use_item_si=args.use_item_si)

    t1 = time.time()
    lm.train(adj_mat, time_mat, cur_time, user_sim_mat, item_sim_mat)
    t2 = time.time()
    users = list(test_user_items_dict.keys())
    user_prediction_items_list = []
    user_truth_items_list = []

    ratings = lm.test()
    for user in users:
        user_truth_items_list.append(test_user_items_dict[user])
        rating = ratings[user]
        rating = torch.from_numpy(rating).view(1, -1)
        user_interacted_items = list(user_interacted_items_dict[user])
        rating[0, user_interacted_items] = -999999.0
        ranked_items = torch.topk(rating, k=max(eval(args.topks)))[1].numpy()[0]
        user_prediction_items_list.append(ranked_items)

    precisions, recalls, f1_scores, mrrs, ndcgs = metric.calculate_all(user_truth_items_list, user_prediction_items_list, eval(args.topks))
    t3 = time.time()
    print('Test metrics:')
    for ind, topk in enumerate(eval(args.topks)):
        print('\t- Top{}:\tF1:{:.4f}\tMRR:{:.4f}\tNDCG:{:.4f}'.format(topk, f1_scores[ind], mrrs[ind], ndcgs[ind]))
    print('Time info:')
    print('\t- Training phase consumes: {:.2f} s'.format(t2 - t1))
    print('\t- Test phase consumes: {:.2f} s'.format(t3 - t2))
    print('\t- Total time consumes: {:.2f} s'.format(t3 - t1))


if __name__=='__main__':
    main()

