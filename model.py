import numpy as np
import scipy.sparse as sp
from sparsesvd import sparsesvd
import multiprocessing


class FIRE(object):
    def __init__(self, init_adj_mat, init_time_mat, decay_factor=0, pri_factor=256, alpha_list=[0.4, 0.2, 0.2, 0.2], use_user_si = True, use_item_si = True):
        self.his_adj_mat = init_adj_mat
        self.his_time_mat = init_time_mat
        self.decay_factor = decay_factor
        self.pri_factor = pri_factor
        self.alpha_list = alpha_list
        self.use_user_si = use_user_si
        self.use_item_si = use_item_si

    def train(self, cur_adj_mat, time_mat, cur_time, user_sim_mat=None, item_sim_mat=None):
        if self.use_user_si:
            rowsum = np.array(user_sim_mat.sum(axis=0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(user_sim_mat)
            norm_adj = norm_adj.dot(d_mat)
            self.user_filter = norm_adj

        if self.use_item_si:
            rowsum = np.array(item_sim_mat.sum(axis=0))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(item_sim_mat)
            norm_adj = norm_adj.dot(d_mat)
            self.item_filter = norm_adj

        his_adj_mat = self.his_adj_mat.toarray()
        his_time_mat = self.his_time_mat.toarray()
        decay_mat = np.exp(self.decay_factor * (his_time_mat - cur_time))
        adj_mat = sp.csr_matrix(decay_mat * his_adj_mat + cur_adj_mat.toarray())
        self.his_adj_mat = sp.csr_matrix(his_adj_mat + cur_adj_mat.toarray())
        self.his_time_mat = sp.csr_matrix(his_time_mat + time_mat.toarray())

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        self.d_mat_i = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        self.d_mat_i_inv = sp.diags(d_inv)
        self.norm_adj = norm_adj.tocsc()
        ut, s, self.vt = sparsesvd(self.norm_adj, self.pri_factor)

    def get_P1(self, batch_test):
        P1 = batch_test.T @ self.user_filter
        res = self.alpha_list[2] * P1.T
        return res

    def get_P2(self, batch_test):
        P2 = batch_test @ self.item_filter
        res = self.alpha_list[3] * P2
        return res

    def get_P3(self, batch_test):
        norm_adj = self.norm_adj
        P31 = batch_test @ norm_adj.T @ norm_adj
        P32 = batch_test @ self.d_mat_i @ self.vt.T @ self.vt @ self.d_mat_i_inv
        P3 = self.alpha_list[0] * P32 + P31
        res = self.alpha_list[1] * P3
        return res

    def test(self):
        adj_mat = self.his_adj_mat
        batch_test = np.array(adj_mat.todense())
        pool=multiprocessing.Pool(processes=8)
        results=[]
        if self.use_user_si:
            results.append(pool.apply_async(func=self.get_P1, args=(batch_test,)))
        if self.use_item_si:
            results.append(pool.apply_async(func=self.get_P2, args=(batch_test,)))
        results.append(pool.apply_async(func=self.get_P3, args=(batch_test,)))
        pool.close()
        pool.join()
        result=0
        for r in results:
            result+=r.get()
        return result
