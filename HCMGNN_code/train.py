# -*- coding: utf-8 -*-
from model import HCMGNN
from Utils.utils_ import *
import warnings

warnings.filterwarnings("ignore")


def Train(train_data, test_data, in_size, args, hg, features):
    np.random.seed(args.seed)
    val_data_pos = test_data[np.where(test_data[:, -1] == 1)]
    '''
    This index is used to calculate the Hit@n, NDCG@n and MRR so that
    the shuffled val set can be redistinguished from the positive or negative samples
    '''
    shuffle_index = np.random.choice(range(len(test_data)), len(test_data), replace=False)
    task_test_data = test_data[shuffle_index]
    model = HCMGNN(
        meta_paths=args.metapaths,
        test_data=val_data_pos,
        in_size=in_size,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        dropout=args.dropout,
        etypes=args.etypes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    myloss = Myloss()
    mrr = MRR()
    matrix = Matrix()
    trainloss = []
    valloss = []
    result_list = []
    hits_max_matrix = np.zeros((1, 3))
    NDCG_max_matrix = np.zeros((1, 3))
    patience_num_matrix = np.zeros((1, 1))
    MRR_max_matrix = np.zeros((1, 1))
    epoch_max_matrix = np.zeros((1, 1))

    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()
        score_train_predict = model(hg, features, train_data)
        train_label = torch.unsqueeze(torch.from_numpy(train_data[:, 3]), 1)
        train_loss = myloss(score_train_predict, train_label, args.loss_gamma)
        trainloss.append(train_loss.item())
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            score_val_predict = model(hg, features, task_test_data)
            val_label = torch.unsqueeze(torch.from_numpy(task_test_data[:, 3]), 1)
            val_label = val_label.to(torch.float)
            val_loss = myloss(score_val_predict, val_label, args.loss_gamma)
            valloss.append(val_loss.item())
            predict_val = np.squeeze(score_val_predict.detach().numpy())
            hits5, ndcg5, sample_hit5, sample_ndcg5 = matrix(5, 30, predict_val, len(val_data_pos), shuffle_index)
            hits3, ndcg3, sample_hit3, sample_ndcg3 = matrix(3, 30, predict_val, len(val_data_pos), shuffle_index)
            hits1, ndcg1, sample_hit1, sample_ndcg1 = matrix(1, 30, predict_val, len(val_data_pos), shuffle_index)
            MRR_num, sample_mrr = mrr(30, predict_val, len(val_data_pos), shuffle_index)
            result = [val_loss.item()] + [hits5] + [hits3] + [hits1] + [ndcg5] + [ndcg3] + [ndcg1] + [MRR_num]
            result_list.append(result)
            if (epoch + 1) % 10 == 0:
                print('Epoch:', epoch + 1, 'Train loss:%.4f' % train_loss.item(),
                      'Val Loss:%.4f' % result_list[epoch][0], 'Hits@5:%.6f' % result_list[epoch][-7],
                      'Hits@3:%.6f' % result_list[epoch][-6], 'Hits@1:%.6f' % result_list[epoch][-5],
                      'NDCG@5:%.6f' % result_list[epoch][-4], 'NDCG@3:%.6f' % result_list[epoch][-3],
                      'NDCG@1:%.6f' % result_list[epoch][-2], 'MRR:%.6f' % result_list[epoch][-1])
            patience_num_matrix = ealy_stop(hits_max_matrix, NDCG_max_matrix, MRR_max_matrix, patience_num_matrix,
                                            epoch_max_matrix,
                                            epoch, hits1, hits3, hits5, ndcg1, ndcg3, ndcg5, MRR_num)
            if patience_num_matrix[0][0] >= args.patience:
                break
    max_epoch = int(epoch_max_matrix[0][0])
    print('Saving train result：', result_list[max_epoch][1:])
    print('the optimal epoch', max_epoch)

    return result_list[max_epoch][1:]
