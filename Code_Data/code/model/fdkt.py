import os
import joblib
import torch
import torch.optim as optimization
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchnet import meter
from tqdm import tqdm
from config import DefaultConfig
from data.data_deal import *
from data.fdktdata import FDKTData
from model.fnn import FNN
from utils import obtain_metrics
from utils import write_csv
from utils.makdir import mkdir

config = DefaultConfig()
Epoch = config.Epoch
batch_size = config.train_batch_size
term_numbers = config.term_numbers
cog_numbers = config.cog_numbers
rule_numbers = config.rule_numbers
weight_decay = config.weight_decay
learning_rate = config.learning_rate
our_dir = config.dir
print_freq = config.print_freq
result_dir = config.result_dir
model_dir = config.model_dir
training_prediction_dir = config.training_prediction_dir
testing_prediction_dir = config.testing_prediction_dir

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
ids = [0, 1]

if config.use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")


def step(fnn_model, input_, isTrain):
    global len_change, max_data_length
    input_scores, input_skills, data_length = Variable(input_[0]), Variable(input_[1]), Variable(input_[2])
    if config.use_gpu:
        max_data_length = torch.Tensor([max(data_length)] * torch.cuda.device_count())
        input_scores = input_scores.to(device)
        input_skills = input_skills.to(device)
        max_data_length = max_data_length.to(device)
    else:
        max_data_length = [max(data_length)]

    batch_target = input_scores[:, 1:]
    max_len = int(max(max_data_length))
    if isTrain:
        if max_len > config.train_len:
            len_change = True
            max_len = config.train_len
        else:
            len_change = False
        T = max_len - 1
    else:
        T = max_len - 1
        len_change = False

    batch_pred, batch_cog = fnn_model(T, input_scores, input_skills, isTrain)

    pred, target, cog = [], [], []
    for b in range(batch_pred.size(0)):
        if len_change:
            dl = min(max_len, data_length[b])
        else:
            dl = data_length[b]
        pred.append(batch_pred[b, : dl - 1])
        target.append(batch_target[b, : dl - 1])
        cog.append(batch_cog[b, : dl - 1, :])
    p, t, c = pred[0], target[0], cog[0]
    for i in range(1, len(pred)):
        p = torch.cat([p, pred[i]])
        t = torch.cat([t, target[i]])
        c = torch.cat([c, cog[i]])
    return p, t, c


def epoch_train(fnn_model, kc_number, train_loader, optimizer, loss_meter, kc_dict, train_max_len):
    target, pred, cog = [], [], []
    for input_ in tqdm(train_loader):
        p, t, c = step(fnn_model, input_, isTrain=True)

        target.append(t.detach().cpu().numpy().tolist())
        pred.append(p.detach().cpu().numpy().tolist())
        cog.append(c.detach().cpu().numpy().tolist())

        criterion = torch.nn.MSELoss(reduction='sum')

        if config.use_gpu:
            criterion.cuda()

        loss = criterion(p, t)
        print(loss)

        loss.requires_grad_(True)
        optimizer.zero_grad()

        torch.autograd.set_detect_anomaly(True)
        loss.backward(retain_graph=True)

        print(loss.item())
        loss_meter.add(loss.item())

        optimizer.step()

    return target, pred, cog


def epoch_predict(fnn_model, data_loader, max_len):
    target, pred, cog = [], [], []
    for input_ in tqdm(data_loader):
        p, t, c = step(fnn_model, input_, isTrain=False)

        target.append(t.detach().cpu().numpy().tolist())
        pred.append(p.detach().cpu().numpy().tolist())
        cog.append(c.detach().cpu().numpy().tolist())
    return target, pred, cog


def collate_fn(batch):
    batch = list(zip(*batch))
    input_scores, input_skills, data_length = batch[0], batch[1], batch[2]
    del batch
    return default_collate(input_scores), default_collate(input_skills), default_collate(data_length)


class FDKT:
    def __init__(self, arg):
        self.arg = arg
        self.path = our_dir + arg[0] + '/' + arg[1]
        self.tr_cc_rmse = [1] * 100
        self.tr_cc_mae = [1] * 100

    def ob_metrics(self, pred, act, epoch):
        rmse, mae = obtain_metrics.obtain_metrics(c_actual=act, c_pred=pred)
        self.tr_cc_rmse[epoch] = rmse
        self.tr_cc_mae[epoch] = mae
        print(str(self.arg[0]) + str(self.arg[1]))
        print('Epoch' + str(epoch))
        return rmse, mae

    def train_and_test(self, arg):
        global start_epoch
        kc_dict, kc_number, train_typeset, test_typeset = get_kc_set(self.path)
        threshold = 1e-5
        if DefaultConfig.code_input:
            if os.path.isfile(self.path + "/coded_train.model"):
                train_data = joblib.load(self.path + "/coded_train.model")
            else:
                train_data = FDKTData(self.path, kc_dict, isTrain=True)
                joblib.dump(filename=self.path + "/coded_train.model", value=train_data)

            if os.path.isfile(self.path + "/coded_test.model"):
                test_data = joblib.load(self.path + "/coded_test.model")
            else:
                test_data = FDKTData(self.path, kc_dict, isTrain=False)
                joblib.dump(filename=self.path + "/coded_test.model", value=test_data)
        else:
            train_data = FDKTData(self.path, kc_dict, isTrain=True)
            test_data = FDKTData(self.path, kc_dict, isTrain=False)

        train_loader = DataLoader(dataset=train_data, batch_size=config.train_batch_size, shuffle=False, drop_last=True,
                                  collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_data, batch_size=config.test_batch_size, shuffle=False, drop_last=True,
                                 collate_fn=collate_fn)
        fnn_model = FNN(term_numbers, cog_numbers, rule_numbers, kc_number, kc_dict, arg, batch_size=batch_size)
        start_epoch = 0
        if config.use_gpu:
            if torch.cuda.device_count() > 1:
                print("Use", torch.cuda.device_count(), 'gpus')
                fnn_model = nn.DataParallel(fnn_model)
        fnn_model.to(device)
        loss_meter = meter.AverageValueMeter()
        optimizer = optimization.Adam(fnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        train_max_len = train_data.max_len
        test_max_len = test_data.max_len
        best_epoch = 0

        for epoch in tqdm(range(start_epoch, Epoch)):
            loss_meter.reset()
            train_target, train_pred, train_cog = epoch_train(fnn_model, kc_number, train_loader, optimizer, loss_meter,
                                                              kc_dict, train_max_len)
            print('Epoch ' + str(epoch) + ' training done.')
            mkdir(DefaultConfig.model_dir)
            mkdir(DefaultConfig.model_dir + str(arg[0]) + '_' + str(arg[1]) + '/')
            joblib.dump(
                filename=DefaultConfig.model_dir + str(arg[0]) + '_' + str(arg[1]) + '/epoch' + str(epoch) + '.model',
                value=fnn_model)
            print(str(arg[0]) + '_' + str(arg[1]) + ': write_txt_ing')
            actual = sum(train_target, [])
            pred = sum(train_pred, [])
            cog = sum(train_cog, [])
            mkdir(DefaultConfig.training_prediction_dir)
            mkdir(DefaultConfig.training_prediction_dir + str(arg[0]) + '_' + str(arg[1]) + '/')
            write_csv.write_performance(actual, type='actual',
                                        dir=DefaultConfig.training_prediction_dir + str(arg[0]) + '_' + str(
                                            arg[1]) + '/epoch' + str(epoch) + '_')
            write_csv.write_performance(pred, type='pred',
                                        dir=DefaultConfig.training_prediction_dir + str(arg[0]) + '_' + str(
                                            arg[1]) + '/epoch' + str(epoch) + '_')
            write_csv.write_performance(cog, type='cog',
                                        dir=DefaultConfig.training_prediction_dir + str(arg[0]) + '_' + str(
                                            arg[1]) + '/epoch' + str(epoch) + '_')
            print(str(arg[0]) + '_' + str(arg[1]) + ': write_txt_success')

            print('metrics calculate...')
            rmse, mae = self.ob_metrics(pred, actual, epoch)
            print('train rmse: ' + str(rmse))
            print('train mae: ' + str(mae))
            if epoch > start_epoch + 1:
                if mae <= self.tr_cc_mae[best_epoch]:
                    best_epoch = epoch

        print('predict for test data...')
        opt_fnn_model = joblib.load(
            filename=DefaultConfig.model_dir + str(arg[0]) + '_' + str(arg[1]) + '/epoch' + str(best_epoch) + '.model')
        test_target, test_pred, test_cog = epoch_predict(opt_fnn_model, test_loader, test_max_len)

        print(str(arg[0]) + '_' + str(arg[1]) + ': write_txt_ing')
        actual = sum(test_target, [])
        pred = sum(test_pred, [])
        cog = sum(test_cog, [])
        mkdir(DefaultConfig.testing_prediction_dir)
        mkdir(DefaultConfig.testing_prediction_dir + str(arg[0]) + '_' + str(arg[1]) + '/')
        write_csv.write_performance(actual, type='actual',
                                    dir=DefaultConfig.testing_prediction_dir + str(arg[0]) + '_' + str(
                                        arg[1]) + '/epoch' + str(best_epoch) + '_')
        write_csv.write_performance(pred, type='pred',
                                    dir=DefaultConfig.testing_prediction_dir + str(arg[0]) + '_' + str(
                                        arg[1]) + '/epoch' + str(best_epoch) + '_')
        write_csv.write_performance(cog, type='cog',
                                    dir=DefaultConfig.testing_prediction_dir + str(arg[0]) + '_' + str(
                                        arg[1]) + '/epoch' + str(best_epoch) + '_')
        print(str(arg[0]) + '_' + str(arg[1]) + ': write_txt_success')

        print('metrics calculate...')
        rmse, mae = self.ob_metrics(pred, actual, best_epoch)
        print('test rmse: ' + str(rmse))
        print('test mae: ' + str(mae))
        mkdir(result_dir)
        mkdir(result_dir + arg[0] + '/')
        mkdir(result_dir + arg[0] + '/' + str(arg[1]) + '/')
        mkdir(result_dir + arg[0] + '/' + str(arg[1]) + '/')

        with open(result_dir + arg[0] + '/' + str(arg[1]) + '/result.txt', 'a') as f:
            f.writelines('best_epoch:' + str(best_epoch) + '\n')
            f.writelines('rmse:' + str(rmse) + '\n')
            f.writelines('mae:' + str(mae) + '\n')

        print("write_result_success")
