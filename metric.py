import numpy as np


def calculate_precision_recall_f1(test_truth_list, test_prediction_list, topk):
    precisions = []
    recalls = []
    f1_scores = []
    for k in topk:
        precision_list = []
        recall_list = []
        for ind, test_truth in enumerate(test_truth_list):
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            precision_dem = k
            recall_dem = len(test_truth_index)
            top_sorted_index = set(test_prediction_list[ind][0:k])
            hit_num = len(top_sorted_index.intersection(test_truth_index))
            precision_list.append(hit_num * 1.0 / (precision_dem + 1e-20))
            recall_list.append(hit_num * 1.0 / (recall_dem + 1e-20))
        precision = np.mean(precision_list)
        recall = np.mean(recall_list)
        f1_score = 2 * precision * recall / (precision + recall + 1e-20)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    return precisions, recalls, f1_scores


def calculate_mrr(test_truth_list, test_prediction_list, topk):
    mrrs = []
    for k in topk:
        mrr_list = []
        for ind, test_truth in enumerate(test_truth_list):
            mrr = 1.0
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][0:k])
            ctr = 1e20
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    ctr = index + 1
                    break
            mrr /= ctr
            mrr_list.append(mrr)
        mrrs.append(np.mean(mrr_list))
    return mrrs


def calculate_ndcg(test_truth_list, test_prediction_list, topk):
    ndcgs = []
    for k in topk:
        ndcg_list = []
        for ind, test_truth in enumerate(test_truth_list):
            dcg = 0
            idcg = 0
            test_truth_index = set(test_truth)
            if len(test_truth_index) == 0:
                continue
            top_sorted_index = set(test_prediction_list[ind][0:k])
            idcg_dem = 0
            for index, itemid in enumerate(top_sorted_index):
                if itemid in test_truth_index:
                    dcg += 1.0 / np.log2(index + 2)
                    idcg += 1.0 / np.log2(idcg_dem + 2)
                    idcg_dem += 1
            ndcg = dcg * 1.0 / (idcg + 1e-20)
            ndcg_list.append(ndcg)
        ndcgs.append(np.mean(ndcg_list))
    return ndcgs


def calculate_all(test_truth_list, test_prediction_list, topk):
    precisions, recalls, f1_scores = calculate_precision_recall_f1(test_truth_list, test_prediction_list, topk)
    mrrs = calculate_mrr(test_truth_list, test_prediction_list, topk)
    ndcgs = calculate_ndcg(test_truth_list, test_prediction_list, topk)
    return precisions, recalls, f1_scores, mrrs, ndcgs
