# -*- coding: utf-8 -*-
import warnings
from data_process import data_lode
from train import Train
from Utils.utils_ import *
warnings.filterwarnings("ignore")



def main_indep():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    features, in_size = data_lode()
    Hits_5, Hits_3, Hits_1, NDCG_5, NDCG_3, NDCG_1, MRR = [list() for x in range(7)]
    train_data_pos = np.array(
        pd.read_csv('./Data/indepent_data/train_data_pos.csv', header=None))
    train_data_neg = np.array(
        pd.read_csv('./Data/indepent_data/train_data_neg.csv',header=None))
    val_data_pos = np.array(
        pd.read_csv('./Data/indepent_data/test_data_pos.csv', header=None))
    val_data_neg = np.array(
        pd.read_csv('./Data/indepent_data/test_data_neg.csv', header=None))
    hg=construct_hg(train_data_pos)
    train_data = np.vstack((train_data_pos, train_data_neg))
    np.random.shuffle(train_data)
    val_data = np.vstack((val_data_pos, val_data_neg))
    result = Train(train_data, val_data, in_size,args, hg,features)
    Hits_5.append(result[0])
    Hits_3.append(result[1])
    Hits_1.append(result[2])
    NDCG_5.append(result[3])
    NDCG_3.append(result[4])
    NDCG_1.append(result[5])
    MRR.append(result[6])
    print('----------independent test finished-----------')
    print('Independent test result：''Hits@5:%.6f' % np.mean(Hits_5), 'Hits@3:%.6f' % np.mean(Hits_3),
                  'Hits@1:%.6f' % np.mean(Hits_1), 'NDCG@5:%.6f' % np.mean(NDCG_5), 'NDCG@3:%.6f' % np.mean(NDCG_3),
                  'NDCG@1:%.6f' % np.mean(NDCG_1),'MRR:%.6f' % np.mean(MRR))
    return np.mean(Hits_5), np.mean(Hits_3),np.mean(Hits_1),np.mean(NDCG_5),np.mean(NDCG_3),np.mean(NDCG_1),np.mean(MRR)

def main_CV():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    features, in_size = data_lode()
    Hits_5, Hits_3, Hits_1, NDCG_5, NDCG_3, NDCG_1, MRR = [list() for x in range(7)]
    fold_num = 0
    for i in range(5):
        fold_num += 1
        train_data_pos = np.array(
            pd.read_csv('./Data/CV_data/CV_' + str(fold_num) + '/train_data_pos.csv', header=None))
        train_data_neg = np.array(
            pd.read_csv('./Data/CV_data/CV_' + str(fold_num) + '/train_data_neg.csv', header=None))
        val_data_pos = np.array(
            pd.read_csv('./Data/CV_data/CV_' + str(fold_num) + '/val_data_pos.csv', header=None))
        val_data_neg = np.array(
            pd.read_csv('./Data/CV_data/CV_' + str(fold_num) + '/val_data_neg.csv',header=None))

        hg=construct_hg(train_data_pos)
        train_data = np.vstack((train_data_pos, train_data_neg))
        np.random.shuffle(train_data)
        val_data = np.vstack((val_data_pos, val_data_neg))
        result = Train(train_data, val_data, in_size,args, hg,features)
        Hits_5.append(result[0])
        Hits_3.append(result[1])
        Hits_1.append(result[2])
        NDCG_5.append(result[3])
        NDCG_3.append(result[4])
        NDCG_1.append(result[5])
        MRR.append(result[6])
    print('----------5 fold CV finished-----------')
    print('5-fold CV result：''Hits@5:%.6f' % np.mean(Hits_5), 'Hits@3:%.6f' % np.mean(Hits_3),
                  'Hits@1:%.6f' % np.mean(Hits_1), 'NDCG@5:%.6f' % np.mean(NDCG_5), 'NDCG@3:%.6f' % np.mean(NDCG_3),
                  'NDCG@1:%.6f' % np.mean(NDCG_1),'MRR:%.6f' % np.mean(MRR))
    return np.mean(Hits_5), np.mean(Hits_3),np.mean(Hits_1),np.mean(NDCG_5),np.mean(NDCG_3),np.mean(NDCG_1),np.mean(MRR)

if __name__ == '__main__':
    args = parameters_set()
    print('Starting the 5-fold CV experiment')
    CV_Hits5, CV_Hits3,CV_Hits1,CV_NDCG_5, CV_NDCG_3, CV_NDCG_1,CV_MRR_num=main_CV()
    with open('./Result/HCMGNN_CV_print.txt', 'a') as f:
        f.write(str(CV_Hits5) + '\t' + str(CV_Hits3) + '\t' + str(CV_Hits1) + '\t' +
            str(CV_NDCG_5) + '\t' + str(CV_NDCG_3) + '\t' + str(CV_NDCG_1) +
            '\t' + str(CV_MRR_num) + '\n')

    print('Starting the independent test experiment')
    indep_Hits5, indep_Hits3,indep_Hits1,indep_NDCG_5, indep_NDCG_3, indep_NDCG_1,indep_MRR_num=main_indep()
    with open('./Result/HCMGNN_indep_print.txt', 'a') as f:
        f.write(str(indep_Hits5) + '\t' + str(indep_Hits3) + '\t' + str(indep_Hits1) + '\t' +
            str(indep_NDCG_5) + '\t' + str(indep_NDCG_3) + '\t' + str(indep_NDCG_1) +
            '\t' + str(indep_MRR_num) + '\n')






