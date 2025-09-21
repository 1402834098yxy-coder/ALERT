import pickle
import numpy as np
import time
from scipy.sparse import csr_matrix

def normalize_matrix(matrix):
    row_means = matrix.mean(axis=1, keepdims=True)
    row_stds = matrix.std(axis=1, keepdims=True)
    for i in range(len(row_stds)):
        if row_stds[i] == 0:
            row_stds[i] = 1
    normalized_matrix = (matrix - row_means) / row_stds
    return normalized_matrix


def preprocess_time(input_dir, feature_number,train_file_names = ['2000-01','2000-02','2000-03','2000-04','2000-05','2000-06','2000-07','2000-08','2000-09','2000-10','2000-11','2000-12',
                    '2001-01','2001-02','2001-03','2001-04','2001-05','2001-06','2001-07','2001-08','2001-09','2001-10','2001-11','2001-12',
                    '2002-01','2002-02','2002-03','2002-04','2002-05','2002-06','2002-07'
                    ]):
    feature_num = feature_number
    file_len = len(train_file_names)

    matrix_vol = np.zeros((file_len, feature_num, feature_num))
    norm_matrix_vol = np.zeros((file_len, feature_num, feature_num))
    train_final_label = []
    k = 0
    query_index = 0
    # initialize a matrix with (feature_num-1, feature_num-1)
    ori_matrix = np.ones((feature_num-1, feature_num-1))
    query_matrix = np.ones((feature_num, feature_num))
    preprocess_time = 0
    for train_file_name in train_file_names:
        input_file = f"{input_dir}/{train_file_name}.pkl"
        print(input_file)
        with open(input_file, "rb") as file:
            
            train_data = pickle.load(file)
            blank_num = 1100000
            test_matrix = np.zeros((feature_num, blank_num))

            first_entries = dict(list(train_data.items())[:feature_num])
           
            print(test_matrix.shape)
            i = 0
            for key in first_entries:
                train_final_label.append(key)
                for length in first_entries[key]["length"]:
                    test_matrix[i, length - 1] = 1
                i += 1
            
            non_empty_columns = np.any(test_matrix != 0, axis=0)

            if np.sum(non_empty_columns) < test_matrix.shape[1]:
                test_matrix = test_matrix[:, non_empty_columns]
                
            print("after removing empty columns: ", test_matrix.shape)
            
            
            mask = np.ones(test_matrix.shape[0], dtype=bool)
            mask[query_index] = False
            query_vec = test_matrix[query_index,:]
            filtered_matrix = test_matrix[mask]
            del test_matrix  # release test_matrix memory

            start_time = time.time()
            result_vec = np.matmul(query_vec, filtered_matrix.T)
            print(result_vec.shape)
            
            vol_query =  np.count_nonzero(query_vec)
            print(vol_query)
            query_matrix[:-1, :-1] = ori_matrix
            # Last column is query_vec (except last element)
            query_matrix[:-1, -1] = result_vec
            # Last row is query_vec (except last element)
            query_matrix[-1, :-1] = result_vec
            # Bottom-right corner is vol_query
            query_matrix[-1, -1] = vol_query

            end_time = time.time()
            preprocess_time += end_time - start_time
            
            
            print("Time taken to load and process the file: ", end_time - start_time)
            
            # result_matrix = np.dot(test_matrix, test_matrix.T)
            # matrix_vol[k] = result_matrix
            k += 1

    for i in range(k):
        start_time = time.time()
        norm_matrix_vol[i] = normalize_matrix(matrix_vol[i])
        end_time = time.time()
        print("Time taken to normalize the matrix: ", end_time - start_time)
        preprocess_time += end_time - start_time
    
    print("all preprocess time: ", preprocess_time)

    print("Done!")
    return preprocess_time