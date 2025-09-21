import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import RidgeClassifier
from catboost import CatBoostClassifier, Pool
import json
import argparse
from sklearn.decomposition import PCA
import time
from simulate_process_time import preprocess_time
import math

def risk_assessment(norm_input_dir1, input_dir1, norm_input_dir2, input_dir2, time_sel, beta, beta_test, delta, eta, if_pca, if_dcluster, gpu_num,model_main_dir,model_pca_dir,dataset_name,preprocess_input_dir):
    time_select_train = math.ceil(time_sel*beta)
    time_select_test = math.ceil(time_sel*beta_test)
    random_select_train = math.ceil(delta*time_select_train)
    random_num_train = math.ceil(eta*time_select_train)
    random_select_test = math.ceil(delta*time_select_test)
    random_num_test = math.ceil(eta*time_select_test)

    rand_num_thres_train = math.comb(time_select_train, random_select_train)
    if random_num_train > rand_num_thres_train:
        random_num_train = rand_num_thres_train
            
    rand_num_thres_test = math.comb(time_select_test, random_select_test)
    if random_num_test > rand_num_thres_test:
        random_num_test = rand_num_thres_test

    # Example usage of the arguments:
    print("Input norm matrix 1 dir:", norm_input_dir1)
    print("Input matrix 1 dir :", input_dir1)
    print("Input norm matrix 2 dir:", norm_input_dir2)
    print("Input matrix 2 dir :", input_dir2)
    print("Time Select train:", time_select_train)
    print("Time Select test:", time_select_test)
    print("Random Select test:", random_select_test)
    print("Random Augmented test:", random_num_test)
    print("Delta:", delta)
    print("Eta:", eta)
    print("If use PCA:", if_pca) 
    print("If use dynamic cluster:", if_dcluster)
    print("GPU Number:", gpu_num)
    
    if dataset_name == 'enron':
        test_file_names = ['20001','20002','20003','20004','20005','20006','20007','20008','20009','200010','200011','200012',
                            '20011','20012','20013','20014','20015','20016','20017','20018','20019','200110','200111','200112',
                            '20021','20022','20023','20024','20025','20026','20027']
    elif dataset_name == 'nytimes' or dataset_name == 'wiki_7000' or dataset_name == 'wiki_3000' or dataset_name == 'wiki_5000':
        test_file_names = ['20001','20002','20003','20004','20005','20006','20007','20008','20009','200010','200011','200012',
                            '20011','20012','20013','20014','20015','20016','20017','20018','20019','200110','200111','200112',
                            '20021','20022','20023','20024','20025','20026']
    else:
        print("dataset is not supported")
        return
              
    # random_select = random_sel
    accuracy_store = []
    test_accuracy_store = []

    warnings.filterwarnings('ignore')
    train_lucene_dict_data = {}
    all_time = 0


    with open(norm_input_dir1, 'rb') as f:
        norm_matrix_vol_train = pickle.load(f)

    with open(input_dir1, 'rb') as f:
        matrix_vol_train = pickle.load(f)

    train_file_len = len(norm_matrix_vol_train)
    print("The length of the train data is ", train_file_len)
    feature_num = len(norm_matrix_vol_train[0])
    print("The feature number of the train data is ", feature_num)

    # resort the norm cooccurrence matrix
    for i in range(train_file_len):
        norm_matrix_vol_train[i] = -np.sort(-norm_matrix_vol_train[i])
    # print(norm_matrix_vol_train[0][0])
      
    matrix_sum_train = np.zeros((feature_num,feature_num))
    for i in range(train_file_len):
        matrix_sum_train += matrix_vol_train[i]

  

    matrix_sum_diag_train = np.diag(matrix_sum_train)
    matrix_sum_diag_sort_train = np.argsort(matrix_sum_diag_train)

    # verse the elements
    matrix_sum_diag_sort_train = matrix_sum_diag_sort_train[::-1]
    # print(matrix_sum_diag_sort_train)



    sort_matrix_vol_train = np.zeros((train_file_len,feature_num,feature_num))
    sort_norm_matrix_vol_train = np.zeros((train_file_len,feature_num,feature_num))

    for i in range(train_file_len):
        for j in range(feature_num):
            sort_matrix_vol_train[i][j] = matrix_vol_train[i][matrix_sum_diag_sort_train[j]]
            sort_norm_matrix_vol_train[i][j] = norm_matrix_vol_train[i][matrix_sum_diag_sort_train[j]]

    with open(f'{model_main_dir}/thres_list.txt', 'r') as f:
        thres_list_str = f.read()
        # Convert string representation of list to actual list of integers
        thres_list = eval(thres_list_str)
    spilt_value  = len(thres_list)-1
    print("The threshold list is ",thres_list)


    with open(norm_input_dir2, 'rb') as f:
        norm_matrix_vol_test = pickle.load(f)

    with open(input_dir2, 'rb') as f:
        matrix_vol_test = pickle.load(f)


    test_file_len = time_select_test
    print("The length of the test data is ", test_file_len)
    feature_num = len(norm_matrix_vol_test[0])
    print("The feature number of the test data is ", feature_num)
    if time_select_test == 1:
        extract_interval = 1
    else:
        extract_interval = (len(matrix_vol_test)-1)// (time_select_test-1)
    print("extract interval is ",extract_interval)
    extract_indices = list(range(len(matrix_vol_test) - 1, -1, -extract_interval))[:test_file_len]
    print("extract indices are ",extract_indices)

   
    # use extract indices to extract the file names
    extract_file_names = [test_file_names[i] for i in extract_indices]
    print("extract file names are ",extract_file_names)
    pre_time = preprocess_time(preprocess_input_dir, feature_num, extract_file_names)
    all_time += pre_time
    print("The preprocess simulation time is ", pre_time)
    
    # extract specific matrix from the matrix_vol and norm_matrix_vol
    
    
    norm_matrix_vol_test = [norm_matrix_vol_test[i] for i in extract_indices]
    matrix_vol_test = [matrix_vol_test[i] for i in extract_indices]
    print("The length of the matrix_vol_test is ",len(matrix_vol_test))
    
    start_time = time.time()
    
    matrix_sum_test = np.zeros((feature_num,feature_num))

    for i in range(test_file_len):
        norm_matrix_vol_test[i] = -np.sort(-norm_matrix_vol_test[i])
        matrix_sum_test += matrix_vol_test[i]
    end_time = time.time()
    print("The time for sorting is ", end_time-start_time)
    all_time += end_time-start_time
       

    matrix_sum_diag_test = np.diag(matrix_sum_test)
    matrix_sum_diag_sort_test = np.argsort(matrix_sum_diag_test)

    # verse the elements
    matrix_sum_diag_sort_test = matrix_sum_diag_sort_test[::-1]
    # print(matrix_sum_diag_sort_test)

    sort_matrix_vol_test = np.zeros((test_file_len,feature_num,feature_num))
    sort_norm_matrix_vol_test = np.zeros((test_file_len,feature_num,feature_num))

    for i in range(test_file_len):
        for j in range(feature_num):
            sort_matrix_vol_test[i][j] = matrix_vol_test[i][matrix_sum_diag_sort_test[j]]
            sort_norm_matrix_vol_test[i][j] = norm_matrix_vol_test[i][matrix_sum_diag_sort_test[j]]

    

    all_final_pred = []
    all_final_ref = []
    all_train_round_acc = 0
    all_test_round_acc = 0
    for i in range(spilt_value):
        # reprocessed_data_train[i] is the data we need to train, it is spilt into 10 groups
        select_number = thres_list[i+1]-thres_list[i]
        print("select number for ",select_number)
        print("round ", i)
        print("index from ",thres_list[i]," to ",thres_list[i+1])
        random_number_train = random_num_train
        random_number_test = random_num_test
        spilt_norm_matrix_train = np.zeros((train_file_len,select_number,feature_num))
        for j in range(train_file_len):
            spilt_norm_matrix_train[j] = sort_norm_matrix_vol_train[j][thres_list[i]:thres_list[i+1]]
        temp_seq_train = np.zeros(feature_num)
        reprocessed_data_train = np.zeros((random_number_train,select_number, feature_num))

        
        random_index_list_train = []
        for l in range(random_number_train):
            random_index = np.random.choice(train_file_len,random_select_train,replace=False)
            random_index_list_train.append(random_index)

        for n in range(select_number):
            for j in range(random_number_train):
                for k in range(random_select_train):
                    temp_index = random_index_list_train[j][k]
                    temp_seq_train += spilt_norm_matrix_train[temp_index][n]
                    # print(temp_seq_train)
                temp_seq_train = temp_seq_train/random_select_train
                reprocessed_data_train[j][n] = temp_seq_train 
                # print("after",temp_seq_train)
                temp_seq_train = np.zeros(feature_num)

        start_time = time.time()
        spilt_norm_matrix_test = np.zeros((test_file_len,select_number,feature_num))
        for j in range(test_file_len):
                spilt_norm_matrix_test[j] = sort_norm_matrix_vol_test[j][thres_list[i]:thres_list[i+1]]
        reprocessed_data_test = np.zeros((random_number_test,select_number, feature_num))
        temp_seq_test = np.zeros(feature_num)

        random_index_list_test = []
        for l in range(random_number_test):
            random_index = np.random.choice(test_file_len,random_select_test,replace=False)
            random_index_list_test.append(random_index)

        for n in range(select_number):
            for j in range(random_number_test):
                for k in range(random_select_test):
                    temp_index = random_index_list_test[j][k]
                    temp_seq_test += spilt_norm_matrix_test[temp_index][n]
                    # print(temp_seq_train)
                temp_seq_test = temp_seq_test/random_select_test
                reprocessed_data_test[j][n] = temp_seq_test 
                # print("after",temp_seq_train)
                temp_seq_test = np.zeros(feature_num)
        
        end_time = time.time()
        print("The time for reprocessing is ", end_time-start_time)
        all_time += end_time-start_time
        
        
        if if_pca == True:
            pca_model_path = f'{model_pca_dir}/{"pca" if if_pca else "nopca"}_{"dcluster" if if_dcluster else "nocluster"}_{thres_list[i]}_{thres_list[i+1]}.pkl'
            pca = pickle.load(open(pca_model_path, 'rb'))
            print("The pca model path is ",pca_model_path)
            start_time = time.time()
            tran_reprocessed_matrix_test = pca.transform(reprocessed_data_test.reshape(-1, feature_num))
            end_time = time.time()
            print("The time for pca is ", end_time-start_time)
            all_time += end_time-start_time



        # test_label = np.arange(select_number)
        # test_label = np.tile(test_label,random_number)
        train_label = matrix_sum_diag_sort_train[thres_list[i]:thres_list[i+1]]
        # print("train_label",train_label)
        train_label = np.tile(train_label,random_number_train)
        test_label = matrix_sum_diag_sort_test[thres_list[i]:thres_list[i+1]]
        # print("test_label",test_label)
        test_label = np.tile(test_label,random_number_test)
        # X_train = tran_reprocessed_matrix_train
        X_test = tran_reprocessed_matrix_test
        y_train = train_label
        y_test = test_label
        # X_train, X_val, y_train, y_val = train_test_split(
        #         X_train, y_train, test_size=0.25, random_state=42)

        



        # load model for testing
        catboost_model = CatBoostClassifier()
        main_model_path = f'{model_main_dir}/cbt_{"pca" if if_pca else "nopca"}_{"dcluster" if if_dcluster else "nocluster"}_{thres_list[i]}_{thres_list[i+1]}.cbm'
        print("The model path is ",main_model_path)
        catboost_model.load_model(main_model_path, format="cbm")
        start_time = time.time()
        y_final_proba = np.zeros((select_number, select_number))
        y_pred_proba = catboost_model.predict_proba(X_test)
        y_pred_labels = np.argmax(y_pred_proba, axis=1)
        # print(y_pred_labels)
        for j in range(select_number):
            for m in range(random_number_test):
                y_final_proba[j] += y_pred_proba[m*select_number+j]
            
        # print(y_final_proba)
        final_final_pred_label = np.zeros(select_number)
        final_pred_label = np.argmax(y_final_proba,axis = 1)
        # print('final predict label',final_pred_label)
        final_label_temp = np.sort(matrix_sum_diag_sort_train[thres_list[i]:thres_list[i+1]])
        for l in range(len(final_pred_label)):
            final_final_pred_label[l] = final_label_temp[final_pred_label[l]]
        # print('final predict label',final_final_pred_label)
        # according to the train label, transport the final_pred_label into the real label

        final_label_ori = matrix_sum_diag_sort_test[thres_list[i]:thres_list[i+1]]
        # print('original label',final_label_ori)
        final_final_pred_label_list = final_final_pred_label.tolist()
        final_label_ori_list = final_label_ori.tolist()
        all_final_pred.extend(final_final_pred_label_list)
        all_final_ref.extend(final_label_ori_list)
        # final_final_pred_label = np.argsort(final_label_temp)
        # final_final_pred_label = np.arange(select_number)
        # print(final_final_pred_label)
        # print(final_pred_label)
        test_accuracy = accuracy_score(final_label_ori, final_final_pred_label)
        all_test_round_acc += test_accuracy
        end_time = time.time()
        print("The time for testing is ", end_time-start_time)
        all_time += end_time-start_time

        print("*******************************************")
        print("round ", i)
        print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
        print("*******************************************")
        result = {}
        result['test_accuracy'] = test_accuracy

    
    acc_thres = 100
    for i in range(int(feature_num/acc_thres)):
        final_acc = accuracy_score(all_final_ref[i*acc_thres:(i+1)*acc_thres], all_final_pred[i*acc_thres:(i+1)*acc_thres])
        print("*******************************************")
        print("final round ", i)
        print("final Accuracy: %.2f%%" % (final_acc * 100.0))
        print("*******************************************")
        
    print("The total time is ", all_time)
    overall_acc = accuracy_score(all_final_ref, all_final_pred)
    print("The overall accuracy is ", overall_acc)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--input_dir1', type=str, help='The first input directory',required=True, default='')
    parser.add_argument('--input_dir2', type=str, help='The second input directory',required=True,default='')
    parser.add_argument('--norm_input_dir1', type=str, help='The first norm input directory',required=True,default='')
    parser.add_argument('--norm_input_dir2', type=str, help='The second norm input directory',required=True,default='')
    parser.add_argument('--time_sel', type=int, help='time selection',required=True,default=1)
    parser.add_argument('--beta', type=float, help='beta',required=True,default=1)
    parser.add_argument('--beta_test', type=float, help='beta_test',required=True,default=1)
    parser.add_argument('--delta', type=float, help='Random selection number',required=True,default=0.4)
    parser.add_argument('--eta', type=float, help='Random augmented number',required=True,default=2)
    parser.add_argument('--if_pca', type=int, help='use or not use PCA',required=True,default=True)
    parser.add_argument('--if_dcluster', type=int, help='dynamic cluster',required=True,default=True)
    parser.add_argument('--gpu_num', type=str, help='gpu',required=True,default=0)
    parser.add_argument('--model_main_dir', type=str, help='model dir',required=True,default='')
    parser.add_argument('--model_pca_dir', type=str, help='model sub dir',required=True,default='')
    parser.add_argument('--dataset_name', type=str, help='dataset name',required=True,default='')
    parser.add_argument('--preprocess_input_dir', type=str, help='preprocess input dir',required=True,default='')
    args = parser.parse_args()
    args.if_pca = bool(args.if_pca)
    args.if_dcluster = bool(args.if_dcluster)
    risk_assessment(args.norm_input_dir1, args.input_dir1, args.norm_input_dir2, args.input_dir2, args.time_sel,args.beta,args.beta_test,args.delta,args.eta,args.if_pca,args.if_dcluster, args.gpu_num,args.model_main_dir,args.model_pca_dir,args.dataset_name,args.preprocess_input_dir)