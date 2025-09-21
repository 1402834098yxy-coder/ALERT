import pickle
import numpy as np
import argparse
import warnings
import os

def normalize_matrix(matrix):
    row_means = matrix.mean(axis=1, keepdims=True)
    row_stds = matrix.std(axis=1, keepdims=True)
    for i in range(len(row_stds)):
        if row_stds[i] == 0:
            row_stds[i] = 1
    normalized_matrix = (matrix - row_means) / row_stds
    return normalized_matrix


def main(input_dir, output_dir_norm, output_dir,dataset,feature_num):
    # Create output directories if they don't exist
    output_dir_norm_path = os.path.dirname(output_dir_norm)
    output_dir_path = os.path.dirname(output_dir)
    
    if not os.path.exists(output_dir_norm_path):
        os.makedirs(output_dir_norm_path)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    # Define the list of file names
    if dataset == 'enron':
        train_file_names = ['20001', '20002', '20003', '20004', '20005', '20006', '20007', '20008', '20009', '200010', '200011', '200012',
                    '20011', '20012', '20013', '20014', '20015', '20016', '20017', '20018', '20019', '200110', '200111', '200112',
                    '20021', '20022', '20023', '20024', '20025', '20026', '20027']
    elif dataset == 'nytimes' or dataset == 'wiki_3000' or dataset == 'wiki_5000' or dataset == 'wiki_7000':
        train_file_names = ['20001', '20002', '20003', '20004', '20005', '20006', '20007', '20008', '20009', '200010', '200011', '200012',
                    '20011', '20012', '20013', '20014', '20015', '20016', '20017', '20018', '20019', '200110', '200111', '200112',
                    '20021' , '20022' , '20023' , '20024' , '20025' , '20026' ]
    else:
        print("dataset not supported")
        return
    
    file_len = len(train_file_names)

    # Initialize arrays and lists to store results
    matrix_vol = np.zeros((file_len, feature_num, feature_num))
    norm_matrix_vol = np.zeros((file_len, feature_num, feature_num))
    train_final_label = []
    k = 0

    # Ignore all warnings
    warnings.filterwarnings("ignore")

    if dataset == 'enron' or dataset == 'nytimes':
        blank_num = 600000
    elif dataset == 'wiki_3000' or dataset == 'wiki_5000' or dataset == 'wiki_7000':
        blank_num = 1100000
    else:
        print("dataset not supported")
        return
    # Load and process each file
    for train_file_name in train_file_names:
        with open(f"{input_dir}/{train_file_name}.pkl", "rb") as file:
            train_data = pickle.load(file)
            test_matrix = np.zeros((feature_num, blank_num))

            # Process the first `feature_num` entries in the dictionary
            first_entries = dict(list(train_data.items())[:feature_num])

            # print(test_matrix.shape)
            i = 0
            for key in first_entries:
                train_final_label.append(key)
                for length in first_entries[key]["length"]:
                    test_matrix[i, length - 1] = 1  # Adjust for 0-based index
                i += 1

            # remove the empty columes
            non_empty_columns = np.any(test_matrix != 0, axis=0)
            test_matrix = test_matrix[:, non_empty_columns]
            
            print(train_file_name, " files after removing empty columns: ", test_matrix.shape)
            
            # Compute the result matrix
            matrix_vol[k] = np.dot(test_matrix, test_matrix.T)
            # matrix_vol[k] = result_matrix
            del test_matrix
            k += 1

    # Normalize the matrices
    for i in range(k):
        norm_matrix_vol[i] = normalize_matrix(matrix_vol[i])

    # Sort matrices in descending order
    # for i in range(k):
    #     norm_matrix_vol[i] = -np.sort(-norm_matrix_vol[i])

    # Save the normalized and original matrices
    with open(output_dir_norm, "wb") as file:
        pickle.dump(norm_matrix_vol, file)
    with open(output_dir, "wb") as file:
        pickle.dump(matrix_vol, file)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and normalize matrix data.")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="The input directory containing the pickle files",
        required=True,
    )
    parser.add_argument(
        "--output_dir_norm",
        type=str,
        help="The output directory for the normalized matrices",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory for the original matrices",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset name",
        required=True,
    )
    parser.add_argument(
        "--feature_num",
        type=int,
        help="The number of features",
        required=True,
        default=3000,
    )
    args = parser.parse_args()
    main(args.input_dir, args.output_dir_norm, args.output_dir,args.dataset,args.feature_num)