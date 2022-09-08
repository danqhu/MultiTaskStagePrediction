from cProfile import label
from ctypes.wintypes import WORD
from functools import lru_cache
from pyexpat import model
from signal import pause
import pandas as pd
import numpy as np
import copy
from utils import feature_preprocessing, imputate_missing_values, one_hot_encoding, feature_normalization
import torch
from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import os
import scipy as sp
import torch.nn as nn
import warnings
import lightgbm
import random
import config
import pickle
import torch.multiprocessing as mp
from gnn import evaluate_gnn, train_gnn, gcn, gat, train_PosNeg_ps_graph, evaluate_PosNeg_ps_graph, train_PosNeg_ps_graph_v2, evaluate_PosNeg_ps_graph_v2
from torch.utils.tensorboard import SummaryWriter



# with open('saved_dictionary.pkl', 'wb') as f:
#     pickle.dump(dictionary, f)
        
# with open('saved_dictionary.pkl', 'rb') as f:
#     loaded_dict = pickle.load(f)


def evaluate_performance(y_true, y_prob, model_name, dataset_name):
    auc = roc_auc_score(y_true=y_true, y_score=y_prob)
    ap = average_precision_score(y_true=y_true, y_score=y_prob)
    # prec = precision_score(y_true=y_true, y_pred=y_pred)
    # reca = recall_score(y_true=y_true, y_pred=y_pred)
    # f1 = f1_score(y_true=y_true, y_pred=y_pred)
    # confu_matr = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print("{}--{}---- AUC: {},  AP:{}".format(dataset_name, model_name, auc, ap))

    return auc, ap

def record_results(df, i, current_method, auc, ap):
    df.loc[df.methods==current_method,'auc_'+str(i)]=auc
    df.loc[df.methods==current_method,'ap_'+str(i)]=ap

def calculate_avg_results(df, current_method):
    aucs = df.iloc[df[df.methods==current_method].index.tolist()[0], 1:11].values
    aps = df.iloc[df[df.methods==current_method].index.tolist()[0], 13:23].values
    df.loc[df.methods==current_method, ['auc_mean','auc_std']] = [np.mean(aucs), np.std(aucs)]
    df.loc[df.methods==current_method, ['ap_mean','ap_std']] = [np.mean(aps), np.std(aps)]

def record_probs(dict, i, current_method, probs, labels):
    '''保存测试集上的预测结果节相应标签，用于绘制全体样本的ROC和PR曲线'''
    if current_method in dict:
        dict[current_method]['fold' + str(i)] = (probs, labels)
    else:
        dict[current_method] = {}
        dict[current_method]['fold' + str(i)] = (probs, labels)

def record_risk_factors(dict, i, current_method, coefs, features):
    if current_method in dict:
        dict[current_method]['fold' + str(i)] = (coefs, features)
    else:
        dict[current_method] = {}
        dict[current_method]['fold' + str(i)] = (coefs, features)

def save_results(current_method, results:pd.DataFrame=None, probs:dict=None, risk_factor:dict=None, random_seed=23):
    
    

    '''调试阶段，暂不保存'''
    # if probs:
    #     with open('results/baseline_probs_' + str(random_seed) + '.pkl', 'wb') as f:
    #         pickle.dump(probs, f)
    #     print('-'*20 + 'Save Probs for {}'.format(current_method) + '-'*20)
    # if risk_factor:
    #     with open('results/baseline_riskfactor_' + str(random_seed) + '.pkl', 'wb') as f:
    #         pickle.dump(risk_factor, f)
    #     print('-'*20 + 'Save Risk_factors for {}'.format(current_method) + '-'*20)

    if not results.empty:
        results.to_csv('results/baseline_results_' + str(random_seed) + '.csv',index=False)
        print('-'*20 + 'Save Results for {}'.format(current_method) + '-'*20)





if __name__ == '__main__':

    svm = False
    rf = False
    lr = False
    lr_l2 = False
    gbdt = False
    mlp = False
    knn = False
    rt = False
    rt_parall = False
    rt_parall_weighted = False
    ps_graph = False
    PosNeg_ps_graph = False
    PosNeg_ps_graph_v2 = False
    PosNeg_ps_graph_v3 = False
    PosNeg_ps_graph_v4 = False


    # svm = True
    # rf = True
    # lr = True
    # lr_l2 = True
    # gbdt = True
    # mlp = True
    # knn = True
    # rt = True
    # rt_parall = True
    # rt_parall_weighted = True
    # ps_graph = True
    # PosNeg_ps_graph = True
    # PosNeg_ps_graph_v2 = True
    # PosNeg_ps_graph_v3 = True
    PosNeg_ps_graph_v4 = True


    current_method = None
    random_seed = config.random_seed
    deepfeatures = True


    if deepfeatures:
        data = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot.csv')
        data_norm = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_normalized.csv')
        data_std = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_standardlized.csv')
    else:
        data = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_nodeep.csv')
        data_norm = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_normalized_nodeep.csv')
        data_std = pd.read_csv(config.local_clinical_data_folder_path + 'data_one_hot_standardlized_nodeep.csv')


    labels = pd.read_csv(config.local_clinical_data_folder_path + 'labels_binary_N2.csv').values.squeeze()
    features_list = data.columns.to_numpy()





    '''results_containers'''
    results = pd.DataFrame(columns=['methods'] + ['auc_'+str(i) for i in range(10)] + ['auc_mean', 'auc_std'] + ['ap_'+str(i) for i in range(10)] + ['ap_mean', 'ap_std'])
    probs_dict = {}
    risk_factor_dict = {}
    model_wts_list = []



    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

        
    if lr:

        current_method = 'lr'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method
        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :]

            labels_test = labels[test_index]
            data_test = data.loc[test_index, :]
            
            estimator = LogisticRegression()
            param_grid = {
                'penalty': ['none'],
                'tol': [1e-4],
                'max_iter': [10000],
                'solver': ['lbfgs'],
                'class_weight': [config.class_weight],
                'random_state': [random_seed]
            }

            gsCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=3)
            gsCV.fit(data_train_val.values, labels_train_val)
            best_estimator = gsCV.best_estimator_

            preds_train = best_estimator.predict(data_train_val.values)
            probs_train = best_estimator.predict_proba(data_train_val.values)[:, 1]
            preds = best_estimator.predict(data_test.values)
            probs = best_estimator.predict_proba(data_test.values)[:, 1]

            
            auc, ap = evaluate_performance(
                y_true=labels_test, 
                y_prob=probs,
                model_name=current_method,
                dataset_name="Test"
            )

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            record_risk_factors(risk_factor_dict, i, current_method, best_estimator.coef_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)


    if lr_l2:

        current_method = 'lr_l1'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method
        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :]

            labels_test = labels[test_index]
            data_test = data.loc[test_index, :]
            
            estimator = LogisticRegression()
            param_grid = {
                'penalty': ['l2'],
                'C': [10, 1, 0.1],
                'tol': [1e-4],
                'max_iter': [10000],
                'solver': ['lbfgs'],
                'class_weight': [config.class_weight],
                'random_state': [random_seed]
            }

            gsCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=3)
            gsCV.fit(data_train_val.values, labels_train_val)
            best_estimator = gsCV.best_estimator_

            preds_train = best_estimator.predict(data_train_val.values)
            probs_train = best_estimator.predict_proba(data_train_val.values)[:, 1]
            preds = best_estimator.predict(data_test.values)
            probs = best_estimator.predict_proba(data_test.values)[:, 1]

            
            auc, ap = evaluate_performance(
                y_true=labels_test, 
                y_prob=probs,
                model_name=current_method,
                dataset_name="Test"
            )

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            record_risk_factors(risk_factor_dict, i, current_method, best_estimator.coef_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)
        

    if svm:
        current_method = 'svm'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method
        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :]

            labels_test = labels[test_index]
            data_test = data.loc[test_index, :]

            # labels_train_val = labels[train_val_index]
            # data_train_val = data_std.loc[train_val_index, :]

            # labels_test = labels[test_index]
            # data_test = data_std.loc[test_index, :]
            
            estimator = SVC()
            param_grid = {
                'C': [10, 1, 0.1],
                # 'gamma': [0.05],
                'kernel': ['rbf'],
                'probability': [True],
                'tol': [1e-5], # for 681
                'class_weight': [None],
                'random_state': [random_seed]
            }

            gsCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=3)
            gsCV.fit(data_train_val.values, labels_train_val)
            best_estimator = gsCV.best_estimator_

            preds_train = best_estimator.predict(data_train_val.values)
            probs_train = best_estimator.predict_proba(data_train_val.values)[:, 1]
            preds = best_estimator.predict(data_test.values)
            probs = best_estimator.predict_proba(data_test.values)[:, 1]

            
            auc, ap = evaluate_performance(
                y_true=labels_test, 
                y_prob=probs,
                model_name=current_method,
                dataset_name="Test"
            )

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            # record_risk_factors(risk_factor_dict, i, current_method, best_estimator.coef_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)


    if rf:
        current_method = 'rf'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method
        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :]

            labels_test = labels[test_index]
            data_test = data.loc[test_index, :]

            estimator = RandomForestClassifier()
            param_grid = {
                'n_estimators': [50, 100, 200],
                'criterion': ['gini'],
                'max_depth': [2, 5],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'random_state': [random_seed],
                'class_weight': [config.class_weight]
            }

            gsCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=3)
            gsCV.fit(data_train_val.values, labels_train_val)
            best_estimator = gsCV.best_estimator_

            preds_train = best_estimator.predict(data_train_val.values)
            probs_train = best_estimator.predict_proba(data_train_val.values)[:, 1]
            preds = best_estimator.predict(data_test.values)
            probs = best_estimator.predict_proba(data_test.values)[:, 1]

            
            auc, ap = evaluate_performance(
                y_true=labels_test, 
                y_prob=probs,
                model_name=current_method,
                dataset_name="Test"
            )

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            record_risk_factors(risk_factor_dict, i, current_method, best_estimator.feature_importances_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)
        

    if gbdt:
        current_method = 'gbdt'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method
        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :]

            labels_test = labels[test_index]
            data_test = data.loc[test_index, :]

            estimator = lightgbm.LGBMClassifier()
            param_grid = {
                'num_leaves': [31],
                'objective': ['binary'],
                'max_depth': [2, 5],
                'subsample': [1.0],
                # 'learning_rate': [1, 0.1, 0.01],
                'n_estimators': [50, 100, 200],
                'class_weight': [config.class_weight],
                'min_child_samples': [1],
                'reg_lambda': [0.01, 0.001],
                # 'reg_alpha': [0.01, 0.001]
            }

            gsCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=3)
            gsCV.fit(data_train_val.values, labels_train_val)
            best_estimator = gsCV.best_estimator_

            preds_train = best_estimator.predict(data_train_val.values)
            probs_train = best_estimator.predict_proba(data_train_val.values)[:, 1]
            preds = best_estimator.predict(data_test.values)
            probs = best_estimator.predict_proba(data_test.values)[:, 1]

            
            auc, ap = evaluate_performance(
                y_true=labels_test, 
                y_prob=probs,
                model_name=current_method,
                dataset_name="Test"
            )

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            record_risk_factors(risk_factor_dict, i, current_method, best_estimator.feature_importances_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)


    if mlp:
        current_method = 'mlp'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method
        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :]

            labels_test = labels[test_index]
            data_test = data.loc[test_index, :]

            estimator = MLPClassifier()
            param_grid = {
                'hidden_layer_sizes': [(100, 500, 100)],
                'activation': ['relu'],
                'alpha': [0.0001],
                'tol': [1e-10],
                'batch_size': ['auto'],
                'max_iter': [500],
                'learning_rate_init': [0.01, 0.001],
                'learning_rate':['adaptive'],
                # 'early_stopping': [True],
                # 'validation_fraction': [0.2],
                # 'n_iter_no_change': [50],
                # 'verbose':[False],
                'random_state': [random_seed]
            }

            gsCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=3)
            gsCV.fit(data_train_val.values, labels_train_val)
            best_estimator = gsCV.best_estimator_

            preds_train = best_estimator.predict(data_train_val.values)
            probs_train = best_estimator.predict_proba(data_train_val.values)[:, 1]
            preds = best_estimator.predict(data_test.values)
            probs = best_estimator.predict_proba(data_test.values)[:, 1]

            
            auc, ap = evaluate_performance(
                y_true=labels_test, 
                y_prob=probs,
                model_name=current_method,
                dataset_name="Test"
            )

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            # record_risk_factors(risk_factor_dict, i, current_method, best_estimator.feature_importances_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)


    if knn:
        
        current_method = 'knn'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method
        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :]

            labels_test = labels[test_index]
            data_test = data.loc[test_index, :]


            estimator = KNeighborsClassifier()
            param_grid = {
                'n_neighbors': [3, 5, 10],
                'p': [1, 2],
                'weights':['distance']
            }

            gsCV = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring='roc_auc', cv=5)
            gsCV.fit(data_train_val.values, labels_train_val)
        
            best_estimator = gsCV.best_estimator_

            preds_train = best_estimator.predict(data_train_val.values)
            probs_train = best_estimator.predict_proba(data_train_val.values)[:, 1]
            preds = best_estimator.predict(data_test.values)
            probs = best_estimator.predict_proba(data_test.values)[:, 1]

            
            auc, ap = evaluate_performance(
                y_true=labels_test, 
                y_prob=probs,
                model_name=current_method,
                dataset_name="Test"
            )

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            # record_risk_factors(risk_factor_dict, i, current_method, best_estimator.coef_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)


    if rt:
        from resnet_transformer import resnet_transformer, resnet_transformer_imgOnly, train_rnt, evaluate_rnt
        
        current_method = 'rt'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method

        data_image_filename = pd.read_csv(config.local_clinical_data_folder_path + 'data_image_filename.csv')


        num_frozen_layer = 400
        resnet_type = '50'
        use_pretrained = True
        hyper_params = {
            'learning_rate': 5e-3,
            'batch_size': 16,
            'num_epochs': 100
        } if deepfeatures else {
            'learning_rate': 5e-3,
            'batch_size': 16,
            'num_epochs': 100
        }
        best_model_dir = './models/rt_'+ ('deep_' if deepfeatures else 'no_deep_') + str(random_seed)+'/'
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        # os.environ['TORCH_HOME']='~/'

        # os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :].drop(columns=[str(i) for i in range(2,10)]) if deepfeatures else data.loc[train_val_index, :]
            data_image_filename_train_val = data_image_filename.loc[train_val_index, :]


            labels_test = labels[test_index]
            data_test = data.loc[test_index, :].drop(columns=[str(i) for i in range(2,10)]) if deepfeatures else data.loc[train_val_index, :]
            data_image_filename_test = data_image_filename.loc[test_index, :]

            
            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
            model_wts = None
            print('Fold {}'.format(i))

            '''训练阶段'''
            for j, (train_index, val_index) in enumerate(skf2.split(np.zeros(len(labels_train_val)), labels_train_val)):
                data_train_val.reset_index(inplace=True, drop=True)
                data_image_filename_train_val.reset_index(inplace=True, drop=True)
                labels_train = labels_train_val[train_index]
                data_train = data_train_val.loc[train_index, :]
                data_image_filename_train = data_image_filename_train_val.loc[train_index, :]
                labels_val = labels_train_val[val_index]
                data_val = data_train_val.loc[val_index, :]
                data_image_filename_val = data_image_filename_train_val.loc[val_index, :]
            
            
                model = resnet_transformer(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
                ) if deepfeatures else resnet_transformer_imgOnly(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
                )
                
                val_auc, val_ap, model_wts = train_rnt(
                    hyper_params, 
                    model, 
                    data_train, 
                    labels_train, 
                    data_image_filename_train, 
                    data_val, 
                    labels_val, 
                    data_image_filename_val, 
                    data_test, 
                    labels_test, 
                    data_image_filename_test
                )
            
            
                break
            
            # path = best_model_dir + 'best_val_model_' + str(i) + '.pt'
            # if not os.path.exists(best_model_dir):
            #     os.makedirs(best_model_dir)
            
            # torch.save({
            #     'model_state_dict': model_wts,
            # }, path)




            '''测试阶段'''
            path = best_model_dir + 'best_val_model_' + str(i) + '.pt'
            best_models = torch.load(path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_test = None
            model_test = resnet_transformer(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
            ) if deepfeatures else resnet_transformer_imgOnly(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
            )

            model_test.load_state_dict(best_models['model_state_dict'])
            model_test.to(device)
            criterian = nn.CrossEntropyLoss()
            test_loss, auc, ap, probs = evaluate_rnt(
                model_test, 
                criterian, 
                data_test, 
                labels_test,
                data_image_filename_test
            )
            print('| test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}'.format(test_loss, auc, ap))

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test)
            # record_risk_factors(risk_factor_dict, i, current_method, best_estimator.coef_, features_list)
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)


    if rt_parall:
        
        from resnet_transformer_parallel import resnet_transformer, resnet_transformer_imgOnly, train_rnt, evaluate_rnt, test_rnt
        current_method = 'rt_parall'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method

        data_image_filename = pd.read_csv(config.local_clinical_data_folder_path + 'data_image_filename.csv')


        num_frozen_layer = 400
        resnet_type = '50'
        use_pretrained = True
        hyper_params = {
            'learning_rate': 1e-2 if deepfeatures else 1e-3,
            'batch_size': 16,
            'num_epochs': 30
        } if deepfeatures else {
            'learning_rate': 1e-2 if deepfeatures else 1e-3,
            'batch_size': 16,
            'num_epochs': 30
        }
        best_model_dir = './models/rt_parallel_'+ ('deep_' if deepfeatures else 'no_deep_') + str(random_seed)+'/'
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        # os.environ['TORCH_HOME']='~/'

        os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29870'
        world_size = torch.cuda.device_count()
        
        # main()

        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :].drop(columns=[str(i) for i in range(2,10)]) if deepfeatures else data.loc[train_val_index, :]
            data_image_filename_train_val = data_image_filename.loc[train_val_index, :]


            labels_test = labels[test_index]
            data_test = data.loc[test_index, :].drop(columns=[str(i) for i in range(2,10)]) if deepfeatures else data.loc[train_val_index, :]
            data_image_filename_test = data_image_filename.loc[test_index, :]

            
            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
            model_wts = None
            print('Fold {}'.format(i))

            # '''训练阶段'''
            # for j, (train_index, val_index) in enumerate(skf2.split(np.zeros(len(labels_train_val)), labels_train_val)):
            #     data_train_val.reset_index(inplace=True, drop=True)
            #     data_image_filename_train_val.reset_index(inplace=True, drop=True)
            #     labels_train = labels_train_val[train_index]
            #     data_train = data_train_val.loc[train_index, :]
            #     data_image_filename_train = data_image_filename_train_val.loc[train_index, :]
            #     labels_val = labels_train_val[val_index]
            #     data_val = data_train_val.loc[val_index, :]
            #     data_image_filename_val = data_image_filename_train_val.loc[val_index, :]
            
            
            #     model = resnet_transformer(
            #         num_classes=2,
            #         num_frozen_layer=num_frozen_layer,
            #         resnet_type=resnet_type,
            #         hidden_feature_size=128,
            #         use_pretrained=use_pretrained
            #     ) if deepfeatures else resnet_transformer_imgOnly(
            #         num_classes=2,
            #         num_frozen_layer=num_frozen_layer,
            #         resnet_type=resnet_type,
            #         hidden_feature_size=128,
            #         use_pretrained=use_pretrained
            #     )
                

            #     mp.spawn(
            #         train_rnt,
            #         args=(
            #             world_size,
            #             hyper_params, 
            #             model, 
            #             data_train, 
            #             labels_train, 
            #             data_image_filename_train, 
            #             data_val, 
            #             labels_val, 
            #             data_image_filename_val, 
            #             data_test, 
            #             labels_test, 
            #             data_image_filename_test,
            #             best_model_dir,
            #             i   # fold number
            #         ),
            #         nprocs=world_size,
            #         join=True
            #     )
                
            
            #     break
            
 


            '''测试阶段'''
            path = best_model_dir + 'best_val_model_' + str(i) + '.pt'
            best_models = torch.load(path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_test = None
            model_test = resnet_transformer(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
            ) if deepfeatures else resnet_transformer_imgOnly(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
            )

            model_test.load_state_dict(best_models['model_state_dict'])
            model_test.to(device)
            criterian = nn.CrossEntropyLoss()
            test_loss, auc, ap, probs, labels_test_new = test_rnt(
                model_test, 
                criterian, 
                data_test, 
                labels_test,
                data_image_filename_test
            )
            print('| test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}'.format(test_loss, auc, ap))

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test_new)  # 此处用labels_test_new，以防测试时样本顺序被shuffle
            # record_risk_factors(risk_factor_dict, i, current_method, best_estimator.coef_, features_list)
        
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)        


    if rt_parall_weighted:
        
        from resnet_transformer_parallel_weighted import resnet_transformer, resnet_transformer_imgOnly, train_rnt, evaluate_rnt, test_rnt
        current_method = 'rt_parall_weighted'
        '''结果表格中为该方法新建一行'''
        results.loc[len(results.index), 'methods'] = current_method

        data_image_filename = pd.read_csv(config.local_clinical_data_folder_path + 'data_image_filename.csv')


        num_frozen_layer = 400
        resnet_type = '50'
        use_pretrained = True
        hyper_params = {
            'learning_rate': 1e-2 if deepfeatures else 1e-3,
            'batch_size': 16,
            'num_epochs': 30
        } if deepfeatures else {
            'learning_rate': 1e-2 if deepfeatures else 1e-3,
            'batch_size': 16,
            'num_epochs': 30
        }
        best_model_dir = './models/rt_parallel_weighted_'+ ('deep_' if deepfeatures else 'no_deep_') + str(random_seed)+'/'
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        # os.environ['TORCH_HOME']='~/'

        os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29870'
        world_size = torch.cuda.device_count()
        
        # main()

        
        for i, (train_val_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):

            labels_train_val = labels[train_val_index]
            data_train_val = data.loc[train_val_index, :].drop(columns=[str(i) for i in range(2,10)]) if deepfeatures else data.loc[train_val_index, :]
            data_image_filename_train_val = data_image_filename.loc[train_val_index, :]


            labels_test = labels[test_index]
            data_test = data.loc[test_index, :].drop(columns=[str(i) for i in range(2,10)]) if deepfeatures else data.loc[train_val_index, :]
            data_image_filename_test = data_image_filename.loc[test_index, :]

            
            skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
            model_wts = None
            print('Fold {}'.format(i))

            '''训练阶段'''
            for j, (train_index, val_index) in enumerate(skf2.split(np.zeros(len(labels_train_val)), labels_train_val)):
                data_train_val.reset_index(inplace=True, drop=True)
                data_image_filename_train_val.reset_index(inplace=True, drop=True)
                labels_train = labels_train_val[train_index]
                data_train = data_train_val.loc[train_index, :]
                data_image_filename_train = data_image_filename_train_val.loc[train_index, :]
                labels_val = labels_train_val[val_index]
                data_val = data_train_val.loc[val_index, :]
                data_image_filename_val = data_image_filename_train_val.loc[val_index, :]
            
            
                model = resnet_transformer(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
                ) if deepfeatures else resnet_transformer_imgOnly(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
                )
                

                mp.spawn(
                    train_rnt,
                    args=(
                        world_size,
                        hyper_params, 
                        model, 
                        data_train, 
                        labels_train, 
                        data_image_filename_train, 
                        data_val, 
                        labels_val, 
                        data_image_filename_val, 
                        data_test, 
                        labels_test, 
                        data_image_filename_test,
                        best_model_dir,
                        i   # fold number
                    ),
                    nprocs=world_size,
                    join=True
                )
                
            
                break
            
 


            '''测试阶段'''
            path = best_model_dir + 'best_val_model_' + str(i) + '.pt'
            best_models = torch.load(path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model_test = None
            model_test = resnet_transformer(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
            ) if deepfeatures else resnet_transformer_imgOnly(
                    num_classes=2,
                    num_frozen_layer=num_frozen_layer,
                    resnet_type=resnet_type,
                    hidden_feature_size=128,
                    use_pretrained=use_pretrained
            )

            model_test.load_state_dict(best_models['model_state_dict'])
            model_test.to(device)
            criterian = nn.CrossEntropyLoss()
            test_loss, auc, ap, probs, labels_test_new = test_rnt(
                model_test, 
                criterian, 
                data_test, 
                labels_test,
                data_image_filename_test
            )
            print('| test loss {:5.2f} | test AUC {:.3f}  | test AP {:.3f}'.format(test_loss, auc, ap))

            record_results(results, i, current_method, auc, ap)
            record_probs(probs_dict, i, current_method, probs, labels_test_new)  # 此处用labels_test_new，以防测试时样本顺序被shuffle
            # record_risk_factors(risk_factor_dict, i, current_method, best_estimator.coef_, features_list)
        
        
        calculate_avg_results(results, current_method)
        save_results(current_method, results, probs_dict, risk_factor_dict, random_seed)        

