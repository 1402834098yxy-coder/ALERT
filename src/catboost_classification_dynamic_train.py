import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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
import math


def train_estimator(norm_input_dir1, input_dir1, norm_input_dir2, input_dir2, time_sel, beta, delta, eta, if_pca, if_dcluster,if_savemodel, if_gpu, gpu_num,model_main_dir,model_pca_dir,dataset_name):
    time_select = math.ceil(time_sel*beta)
    random_select = math.ceil(delta*time_select)
    random_num = math.ceil(eta*time_select) 

    # Example usage of the arguments:
    print("Input norm matrix 1 dir:", norm_input_dir1)
    print("Input matrix 1 dir :", input_dir1)
    print("Input norm matrix 2 dir:", norm_input_dir2)
    print("Input matrix 2 dir :", input_dir2)
    print("Time Select:", time_sel)
    print("Random Select:", random_select)
    print("Random Augmented:", random_num)
    print("If use PCA:", if_pca) 
    print("If use dynamic cluster:", if_dcluster)
    print("If save model:", if_savemodel)
    print("If use GPU:", if_gpu)
    print("GPU Number:", gpu_num)
    print("Dataset:", dataset_name)

    accuracy_store = []
    test_accuracy_store = []

    warnings.filterwarnings('ignore')

   
    training_time = 0
    testing_time = 0


    with open(norm_input_dir1, 'rb') as f:
        norm_matrix_vol_train = pickle.load(f)

    with open(input_dir1, 'rb') as f:
        matrix_vol_train = pickle.load(f)

    file_len = len(norm_matrix_vol_train)
    print("The length of the train data is ", file_len)
    feature_num = len(norm_matrix_vol_train[0])
    print("The feature number of the train data is ", feature_num)



    start_train_time = time.time()

    # resort the norm cooccurrence matrix
    for i in range(file_len):
        norm_matrix_vol_train[i] = -np.sort(-norm_matrix_vol_train[i])
    # print(norm_matrix_vol_train[0][0])
      
    matrix_sum_train = np.zeros((feature_num,feature_num))
    for i in range(file_len):
        matrix_sum_train += matrix_vol_train[i]



    matrix_sum_diag_train = np.diag(matrix_sum_train)

    matrix_sum_diag_sort_train = np.argsort(matrix_sum_diag_train)

    # verse the elements
    matrix_sum_diag_sort_train = matrix_sum_diag_sort_train[::-1]
    # print(matrix_sum_diag_sort_train)

    sort_matrix_vol_train = np.zeros((file_len,feature_num,feature_num))
    sort_norm_matrix_vol_train = np.zeros((file_len,feature_num,feature_num))

    for i in range(file_len):
        for j in range(feature_num):
            sort_matrix_vol_train[i][j] = matrix_vol_train[i][matrix_sum_diag_sort_train[j]]
            sort_norm_matrix_vol_train[i][j] = norm_matrix_vol_train[i][matrix_sum_diag_sort_train[j]]

    if if_dcluster == True:
    # sort all diag1 
        print("use dynamic cluster")
        all_diag1 = np.zeros((file_len,feature_num))
        for i in range(file_len):
            diag1 = np.diag(matrix_vol_train[i]) # attension: the diag1 is sorted with the volume of the matrix
            sorted_indices1 = np.argsort(-diag1)
            q_ranks = np.empty_like(sorted_indices1)
            # Map each index back to its rank
            for rank, idx in enumerate(sorted_indices1):
                q_ranks[idx] = rank   # +1 to make rank 1-based if needed
            all_diag1[i] = q_ranks
        std_diag1 = np.std(all_diag1, axis=0)
        mean_diag1 = np.mean(all_diag1, axis=0)
        decision_list = []
        threshold_list = [50, 100, 200, 500]
        sign = True
        thres_list = [0]
        scale_factor = 5
        while sign:
            for i in range(len(threshold_list)):
                threshold = threshold_list[i]
                avg_std = 0
                avg_mean = 0

                for j in range(threshold):
                    avg_std += std_diag1[j]
                    avg_mean += mean_diag1[j]

                avg_std /= threshold
                avg_mean /= threshold

                lower_bound = avg_mean - scale_factor*avg_std
                upper_bound = avg_mean + scale_factor*avg_std
                if lower_bound >= thres_list[-1] and upper_bound <= thres_list[-1] +threshold:
                    # print(diff_thres)
                    decision_list.append(threshold)

                    temp_thres = thres_list[-1] + decision_list[-1]
                    thres_list.append(temp_thres)
                    std_diag1 = std_diag1[threshold:]
                    mean_diag1 = mean_diag1[threshold:]
                    break
                # maximum threshold is 1000, avoid unbounded loop
                elif threshold == 500:
                    decision_list.append(threshold)

                    temp_thres = thres_list[-1] + decision_list[-1]
                    thres_list.append(temp_thres)
                    std_diag1 = std_diag1[threshold:]
                    mean_diag1 = mean_diag1[threshold:]
                    break
                else:
                    continue
            
            if decision_list and len(std_diag1) <= decision_list[-1]:
                thres_list[-1] -= decision_list[-1]
                decision_list[-1] += len(std_diag1)

                temp_thres = thres_list[-1] + decision_list[-1]
                thres_list.pop()
                thres_list.append(temp_thres)

                sign = False
                break
    else: # if not use dynamic cluster
        print("not use dynamic cluster")
        thres_list = [0,feature_num]
    spilt_value  = len(thres_list)-1
    print("The threshold list is ",thres_list)
    
    end_train_time = time.time()
    training_time += end_train_time - start_train_time
    print("training data process time is ", training_time)

    with open(norm_input_dir2, 'rb') as f:
        norm_matrix_vol_test = pickle.load(f)

    with open(input_dir2, 'rb') as f:
        matrix_vol_test = pickle.load(f)
        
    start_time = time.time()
    for i in range(file_len):
        norm_matrix_vol_test[i] = -np.sort(-norm_matrix_vol_test[i])
    
    matrix_sum_test = np.zeros((feature_num,feature_num))
    for i in range(file_len):
        matrix_sum_test += matrix_vol_test[i]


    matrix_sum_diag_test = np.diag(matrix_sum_test)
    matrix_sum_diag_sort_test = np.argsort(matrix_sum_diag_test)

    # verse the elements
    matrix_sum_diag_sort_test = matrix_sum_diag_sort_test[::-1]
    # print(matrix_sum_diag_sort_test)

    sort_matrix_vol_test = np.zeros((file_len,feature_num,feature_num))
    sort_norm_matrix_vol_test = np.zeros((file_len,feature_num,feature_num))

    for i in range(file_len):
        for j in range(feature_num):
            sort_matrix_vol_test[i][j] = matrix_vol_test[i][matrix_sum_diag_sort_test[j]]
            sort_norm_matrix_vol_test[i][j] = norm_matrix_vol_test[i][matrix_sum_diag_sort_test[j]]

    end_time = time.time()
    preprocess_time = end_time - start_time
    testing_time += preprocess_time
    print("testing data process time is ", preprocess_time)


    all_final_pred = []
    all_final_ref = []
    all_train_round_acc = 0
    all_test_round_acc = 0
    for i in range(spilt_value):
        start_train_time = time.time()
        # reprocessed_data_train[i] is the data we need to train, it is spilt into 10 groups
        select_number = thres_list[i+1]-thres_list[i]
        print("select number for ",select_number)
        print("round ", i)
        print("index from ",thres_list[i]," to ",thres_list[i+1])
        spilt_norm_matrix_train = np.zeros((file_len,select_number,feature_num))
        for j in range(file_len):
            spilt_norm_matrix_train[j] = sort_norm_matrix_vol_train[j][thres_list[i]:thres_list[i+1]]
        random_number = random_num
        temp_seq_train = np.zeros(feature_num)
        reprocessed_data_train = np.zeros((random_number,select_number, feature_num))

        
        random_index_list_train = []
        for l in range(random_number):
            random_index = np.random.choice(file_len,random_select,replace=False)
            random_index_list_train.append(random_index)

        for n in range(select_number):
            for j in range(random_number):
                for k in range(random_select):
                    temp_index = random_index_list_train[j][k]
                    temp_seq_train += spilt_norm_matrix_train[temp_index][n]
                    # print(temp_seq_train)
                temp_seq_train = temp_seq_train/random_select
                reprocessed_data_train[j][n] = temp_seq_train 
                # print("after",temp_seq_train)
                temp_seq_train = np.zeros(feature_num)

        start_time = time.time()
        spilt_norm_matrix_test = np.zeros((file_len,select_number,feature_num))
        for j in range(file_len):
                spilt_norm_matrix_test[j] = sort_norm_matrix_vol_test[j][thres_list[i]:thres_list[i+1]]
        reprocessed_data_test = np.zeros((random_number,select_number, feature_num))
        temp_seq_test = np.zeros(feature_num)

        random_index_list_test = []
        for l in range(random_number):
            random_index = np.random.choice(file_len,random_select,replace=False)
            random_index_list_test.append(random_index)

        for n in range(select_number):
            for j in range(random_number):
                for k in range(random_select):
                    temp_index = random_index_list_test[j][k]
                    temp_seq_test += spilt_norm_matrix_test[temp_index][n]
                    # print(temp_seq_train)
                temp_seq_test = temp_seq_test/random_select
                reprocessed_data_test[j][n] = temp_seq_test 
                # print("after",temp_seq_train)
                temp_seq_test = np.zeros(feature_num)
        end_time = time.time()
        augment_time = end_time - start_time
        testing_time += augment_time
        print("testing data augment time is ", augment_time)
        
        if if_pca == True:
            group_data = reprocessed_data_train
            flattern_data = group_data.reshape(-1, feature_num)
            pca = PCA(n_components=0.995)


            tran_reprocessed_matrix_train = pca.fit_transform(flattern_data)
            print("The shape of the train data is ", tran_reprocessed_matrix_train.shape)


            tran_reprocessed_matrix_test = pca.transform(reprocessed_data_test.reshape(-1, feature_num))
            n_components_actual = tran_reprocessed_matrix_train.shape[1]
            if n_components_actual < 20 or n_components_actual > 50:
                n_components_to_use = max(20, min(n_components_actual, 50))
                pca = PCA(n_components=n_components_to_use)
                tran_reprocessed_matrix_train = pca.fit_transform(flattern_data)
                start_time = time.time()
                tran_reprocessed_matrix_test = pca.transform(reprocessed_data_test.reshape(-1, feature_num))
                end_time = time.time()
                pca_time = end_time - start_time
                testing_time += pca_time
                print("PCA time is ", pca_time)
                
                print(f"Adjusted the number of components to ensure between 5 and 50 features. Using {n_components_to_use} components.")
                print("The new shape of the train data is ", tran_reprocessed_matrix_train.shape)
            
            if if_savemodel == True:
                pca_model_path = f'{model_pca_dir}/{"pca" if if_pca else "nopca"}_{"dcluster" if if_dcluster else "nocluster"}_{thres_list[i]}_{thres_list[i+1]}.pkl'
                pickle.dump(pca, open(pca_model_path, 'wb'))
        
        else:
            tran_reprocessed_matrix_train = reprocessed_data_train.reshape(random_number*select_number, feature_num)
            tran_reprocessed_matrix_test = reprocessed_data_test.reshape(random_number*select_number, feature_num)

        train_label = matrix_sum_diag_sort_train[thres_list[i]:thres_list[i+1]]
        # print("train_label",train_label)
        train_label = np.tile(train_label,random_number)
        test_label = matrix_sum_diag_sort_test[thres_list[i]:thres_list[i+1]]
        # print("test_label",test_label)
        test_label = np.tile(test_label,random_number)
        X_train = tran_reprocessed_matrix_train
        X_test = tran_reprocessed_matrix_test
        y_train = train_label
        y_test = test_label
        X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.25, random_state=42)

        if if_gpu == True:
            catboost_model = CatBoostClassifier(
                iterations=100,
                learning_rate=1e-1,
                depth=6,  
                l2_leaf_reg=3,  
                border_count=22,  
                random_strength=1,  
                bagging_temperature=0.8, 
                min_data_in_leaf=5, 
                task_type='GPU',
                devices=str(gpu_num),
                loss_function='MultiClass',
                auto_class_weights='Balanced',
                early_stopping_rounds=5
            )
        else:
             catboost_model = CatBoostClassifier(
                iterations=100,
                learning_rate=1e-1,
                depth=6,  
                l2_leaf_reg=3,  
                border_count=22,  
                random_strength=1,  
                bagging_temperature=0.8, 
                min_data_in_leaf=5, 
                loss_function='MultiClass',
                auto_class_weights='Balanced',
                early_stopping_rounds=5
            )




        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)


        catboost_model.fit(
            train_pool,
            eval_set=val_pool,
            verbose=False  
        )
        end_train_time = time.time()
        training_time += end_train_time - start_train_time
        print(f"training time until {thres_list[i+1]} keyword is ", training_time)
        # save catboost model
        if if_savemodel == True:
            print("save model")
            thres_list_path = f'{model_main_dir}/thres_list.txt'
            with open(thres_list_path, 'w') as f:
                f.write(str(thres_list))

            main_model_path = f'{model_main_dir}/cbt_{"pca" if if_pca else "nopca"}_{"dcluster" if if_dcluster else "nocluster"}_{thres_list[i]}_{thres_list[i+1]}.cbm'
            catboost_model.save_model(main_model_path, format="cbm", export_parameters=None)
        y_pred = catboost_model.predict(X_val)
        # print(y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        all_train_round_acc += accuracy
        accuracy_store.append(accuracy)


        y_final_proba = np.zeros((select_number, select_number))
        start_test_time = time.time()

        y_pred_proba = catboost_model.predict_proba(X_test)
        y_pred_labels = np.argmax(y_pred_proba, axis=1)
        # print(y_pred_labels)
        for j in range(select_number):
            for m in range(random_number):
                y_final_proba[j] += y_pred_proba[m*select_number+j]
            
        # print(y_final_proba)
        final_final_pred_label = np.zeros(select_number)
        final_pred_label = np.argmax(y_final_proba,axis = 1)
        # print('final predict label',final_pred_label)
        final_label_temp = np.sort(matrix_sum_diag_sort_train[thres_list[i]:thres_list[i+1]])
        for l in range(len(final_pred_label)):
            final_final_pred_label[l] = final_label_temp[final_pred_label[l]]

        final_label_ori = matrix_sum_diag_sort_test[thres_list[i]:thres_list[i+1]]
        # print('original label',final_label_ori)
        final_final_pred_label_list = final_final_pred_label.tolist()
        final_label_ori_list = final_label_ori.tolist()
        all_final_pred.extend(final_final_pred_label_list)
        all_final_ref.extend(final_label_ori_list)
        test_accuracy = accuracy_score(final_label_ori, final_final_pred_label)
        
        end_test_time = time.time()
        testing_time += end_test_time - start_test_time
        print(f"testing time until {thres_list[i+1]} keyword is ", testing_time)
        
        all_test_round_acc += test_accuracy

        print("*******************************************")
        print("round ", i)
        print("Train Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Test Accuracy: %.2f%%" % (test_accuracy * 100.0))
        print("*******************************************")
        #put the result of the model into the json file, all result should be put in one file
        #write the experiment result into a json file
        result = {}
        result['train_accuracy'] = accuracy
        result['test_accuracy'] = test_accuracy

    
    acc_thres = 100
    for i in range(int(feature_num/acc_thres)):
        final_acc = accuracy_score(all_final_ref[i*acc_thres:(i+1)*acc_thres], all_final_pred[i*acc_thres:(i+1)*acc_thres])
        print("*******************************************")
        print("final round ", i)
        print("final Accuracy: %.2f%%" % (final_acc * 100.0))
        print("*******************************************")

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
    parser.add_argument('--delta', type=float, help='Random selection number',required=True,default=0.4)
    parser.add_argument('--eta', type=float, help='Random augmented number',required=True,default=2)
    parser.add_argument('--if_pca', type=int, help='PCA',required=True,default=1)
    parser.add_argument('--if_dcluster', type=int, help='dynamic cluster',required=True,default=1)
    parser.add_argument('--if_savemodel', type=int, help='save model',required=True,default=0)
    parser.add_argument('--if_gpu', type=int, help='gpu',required=True,default=0)
    parser.add_argument('--gpu_num', type=str, help='gpu',required=True,default=0)
    parser.add_argument('--model_main_dir', type=str, help='model main dir',required=True,default='')
    parser.add_argument('--model_pca_dir', type=str, help='model pca dir',required=True,default='')
    parser.add_argument('--dataset_name', type=str, help='dataset',required=True,default='enron')
    args = parser.parse_args()
    args.if_pca = bool(args.if_pca)
    args.if_dcluster = bool(args.if_dcluster)
    args.if_savemodel = bool(args.if_savemodel)
    args.if_gpu = bool(args.if_gpu)
    train_estimator(args.norm_input_dir1, args.input_dir1, args.norm_input_dir2, args.input_dir2, args.time_sel,args.beta,args.delta,args.eta,args.if_pca,args.if_dcluster,args.if_savemodel, args.if_gpu,args.gpu_num,args.model_main_dir,args.model_pca_dir,args.dataset_name)

