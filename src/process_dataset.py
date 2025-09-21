import os
import random
import re
import nltk
import time
import email
import pickle
import mailbox
import datetime as dt
import numpy as np
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import defaultdict
from tqdm import tqdm
from email.utils import parsedate_to_datetime
import argparse
from Processed_Dict import (RAW_DATA_PATH, PRO_DATA_PATH, CLASSIFY_PATH, KEYWORDS_PATH, DESIRED_YEARS, TRANSFER_PATH,
                                          EXTRACT_RAW_PATH, EXTRACT_DIVISION_PATH, EXTRACT_ACCUMULATION_PATH,
                                          DIVISION_PATH, DIVISION_PROCESSED_PATH, ACCUMULATION_PATH, ACCUMULATION_PROCESSED_PATH, colors)



# initialize frequency
# def read_frequency_from_file():
#     file_path_1 = f'../dataset/bow-nytimes.pkl'
#     # file_path_1 = f'../dataset/enron-full_stems_to_words.pkl'
#     with open(file_path_1, 'rb') as f:
#         loaded_data_1 = pickle.load(f)
#     file_path_2 = f'../dataset/enron_db.pkl'
#     # file_path_2 = f'../dataset/nytimes_db.pkl'
#     with open(file_path_2, 'rb') as f:
#         loaded_data_2 = pickle.load(f)
#     frequency = {}
#     # print(list(loaded_data_1.keys()))
#     # print(list(loaded_data_2[1].values()))
#     for word, count_trend in zip(list(loaded_data_1[2].keys()), list(loaded_data_2[1].values())):
#     # for word in list(loaded_data_1[1]):
#     #     frequency_i = {'count': 0, 'trend': count_trend['trend']}
#         trend_i = count_trend['trend']
#         frequency_i = {'count': 0, 'trend': np.zeros_like(trend_i)}
#         frequency[word] = frequency_i
#     return frequency


def save_to_file(dataset_words, dataset_date, dataset_index, path):
    names = ['dataset_words', 'dataset_date', 'dataset_index']
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, names[0] + '.pkl'), 'wb') as f:
        pickle.dump(dataset_words, f)
    with open(os.path.join(path, names[1] + '.pkl'), 'wb') as f:
        pickle.dump(dataset_date, f)
    with open(os.path.join(path, names[2] + '.pkl'), 'wb') as f:
        pickle.dump(dataset_index, f)
    print(colors['cyan'].format('Dataset Save √'))


# read dataset_words, dataset_date, dataset_index from path
def read_from_file(path):
    names = ['dataset_words', 'dataset_date', 'dataset_index']
    with open(os.path.join(path, names[0] + '.pkl'), 'rb') as f:
        dataset_words = pickle.load(f)
    with open(os.path.join(path, names[1] + '.pkl'), 'rb') as f:
        dataset_date = pickle.load(f)
    with open(os.path.join(path, names[2] + '.pkl'), 'rb') as f:
        dataset_index = pickle.load(f)
    print(colors['cyan'].format('Dataset read √'))
    return dataset_words, dataset_date, dataset_index


# read keywords from file
def read_keywords_from_file(dataset_name):
    file_path = KEYWORDS_PATH[dataset_name]
    with open(file_path, 'rb') as f:
        keywords = pickle.load(f)
    print(colors['cyan'].format('Keywords load √'))
    if dataset_name == 'nytimes':
        return keywords[2]['stems_to_words']
    else:
        return keywords


# decode the payload
def self_decode(payload, charset='latin-1'):
    """
    parameters:
    payload -- the payload to decode
    charset -- the charset to decode, default is 'latin-1'
    return:
    the decoded string. if the decoding process fails, it will use 'latin-1' to decode and replace the undecoded characters with '?'
    exception:
    LookupError -- when the specified charset is not found
    TypeError -- when the payload is not a bytes type
    """
    try:
        return payload.decode(charset, errors='replace')
    except (LookupError, TypeError):
        return payload.decode('latin-1', errors='replace')


# process the message, extract all pure text content, return the concatenated string of all pure text content.
def process_email(message):
    payload = []
    for part in message.walk():
        if part.get_content_type() == 'text/plain':
            payload.append(self_decode(part.get_payload(decode=True), part.get_content_charset() or 'latin-1'))
    payload = ''.join(payload)
    return payload


# extract the date from the message, and format it as year-month.
def extract_date(message):
    date_header = message.get('Date')
    if date_header:
        email_date = parsedate_to_datetime(date_header)
        year_month = email_date.strftime("%Y-%m")
        return year_month


# load the dates from the corresponding date file of dataset_name.
def get_dates(dataset_name):
    with open(os.path.join(CLASSIFY_PATH[dataset_name], 'dates.pkl'), 'rb') as f:
        dates = pickle.load(f)
    print(colors['cyan'].format('Dates get √'))
    return dates


# format the date, the original date is 'YYYY-MM', format it as 'YYYYM'
def format_date(date):
    date_year = date.split('-')[0]
    date_month = date.split('-')[1]
    date_month = int(date_month)
    date = date_year + str(date_month)
    return date


# generate the dates within the specified time period, and save them to a file.
def date_loop(dateset_name='lucene', start_year=2002, end_year=2020):
    """
    parameters:
    dataset_name -- the name of the dataset, used to determine the path to save the dates, default is 'lucene'
    start_year -- the start year, default is 2002
    end_year -- the end year, default is 2020
    return:
    file_names -- the list of file names generated by the dates
    dates -- the list of dates generated
    """
    start_date = dt.date(start_year, 1, 1)
    end_date = dt.date(end_year, 12, 1)
    if dateset_name == 'nytimes' or dateset_name == 'wiki_3000' or dateset_name == 'wiki_5000' or dateset_name == 'wiki_7000':
        end_date = dt.date(end_year, 6, 1)
    if dateset_name == 'enron':
        end_date = dt.date(end_year, 7, 1)
    current_date = start_date
    dates = []
    file_names = []
    while current_date <= end_date:
        year_month = current_date.strftime('%Y-%m')
        file_name = year_month.split('-')[0] + year_month.split('-')[1]
        file_names.append(file_name)
        dates.append(year_month)
        year = current_date.year + (current_date.month // 12)
        month = current_date.month % 12 + 1
        current_date = dt.date(year, month, 1)
    os.makedirs(CLASSIFY_PATH[dateset_name], exist_ok=True)
    with open(os.path.join(CLASSIFY_PATH[dateset_name], 'dates.pkl'), 'wb') as f:
        pickle.dump(dates, f)
    print(colors['cyan'].format('Dates save √'))
    return file_names, dates


def print_str(phrase, dataset_name, part='train', mark_part=False, mark_force=False):
    phrase = colors['green'].format(phrase)
    high_dataset_name = colors['purple'].format(dataset_name)
    high_part = colors['blue'].format(part)
    if not mark_force:
        phrase += " for dataset {:s}".format(high_dataset_name) + " of {:s}".format(high_part) if mark_part else ""
        phrase += " is already done."
        print(phrase)
    else:
        phrase += " for dataset {:s}".format(high_dataset_name) + " of {:s}.".format(high_part) if mark_part else " ."
        print(phrase)


def clearn_email(keywords_read, message):
    unique_words = list(set(re.findall(r'\w+', message)))
    unique_words = list(set([word.lower() for word in unique_words if word.isalpha()]))
    english_stopwords = stopwords.words('english')
    valid_unique_words = [word if (word.lower() not in english_stopwords and 2 < len(word) < 20 and word.isalpha())
                          else None for word in unique_words]
    valid_unique_words = sorted(list(set(valid_unique_words) - {None}))
    replace_map = {}
    for key, values in keywords_read.items():
        for value in values:
            replace_map[value] = key
    valid_unique_keywords = [replace_map[word] for word in valid_unique_words if word in replace_map]
    valid_unique_keywords_set = set(valid_unique_keywords)
    valid_unique_keywords_list = list(valid_unique_keywords_set)
    valid_unique_numbers_list = []
    keywords_enron = list(keywords_read.keys())
    indexes = list(range(len(keywords_enron)))
    key_number_dict = dict(zip(keywords_enron, indexes))
    for keyword in valid_unique_keywords_list:
        valid_unique_numbers_list.append(key_number_dict[keyword])
    return valid_unique_numbers_list

def apply_countermeasures(dataset_words, countermeasure, keyword_size):
    keyword_size = keyword_size
    # countermeasures =['padding_seal']
    # countermeasures = ['padding_linear']

    dataset_array = np.zeros((len(dataset_words),keyword_size))
    # fill the matrix with the dataset_words
    # write each dataset_words to the matrix
    for i in tqdm(range(len(dataset_words))):
        valid_words = [word for word in dataset_words[i] if int(word) < keyword_size]
        for word in valid_words:
            dataset_array[i, int(word)] = 1
        # for word in dataset_words[i]:
        #     dataset_array[i, int(word)] = 1
    # shape of dataset_array is (len(dataset_words),keyword_size)
    # shape of dataset_array_t is (keyword_size,len(dataset_words))
    dataset_array_t = dataset_array.T
    dataset_words_new = []
    
    if countermeasure == 'padding_seal':
        print("Countermeasure is ",countermeasure)
        n=2
        if n==0 or n==1:
            # print(dataset_array_t.shape)
            for i in tqdm(range(len(dataset_array))):
                word_indices = np.nonzero(dataset_array[i])[0]
                dataset_words_new.append(word_indices.tolist())
                return dataset_words_new
        
        
        query_number = keyword_size

        padded_query_d = np.zeros((query_number,0))
        for i in tqdm(range(query_number)):
            s = np.count_nonzero(dataset_array_t[i])
            power = 0
            while(1):
                if s > (n**power):
                    power += 1
                else:
                    padding_number = (n**power)-s
                    break
            # print(padding_number)
            # zeros_count = np.count_nonzero(~dataset_array_t[i])
            if padding_number > len(padded_query_d[0]):
                pad_doc = np.zeros((query_number, padding_number - len(padded_query_d[0])))
                padded_query_d = np.hstack((padded_query_d,pad_doc))
                temp = np.ones((1,padding_number))
                padded_query_d[i] = temp
            else:
                temp = np.zeros(len(padded_query_d[0]))
                index = np.where(temp==0)[0]
                
                pad_index = np.random.choice(index,padding_number,replace=False)
                temp[pad_index] = 1
                padded_query_d[i] = padded_query_d[i] + temp

        padded_query_d = np.hstack((dataset_array_t,padded_query_d))
        padded_query_d = padded_query_d.T
        for i in tqdm(range(len(padded_query_d))):
            word_indices = np.nonzero(padded_query_d[i])[0]
            dataset_words_new.append(word_indices.tolist())
        return dataset_words_new

    elif countermeasure == 'padding_linear':
        print("Countermeasure is ",countermeasure)
        n=500
        if n==0 or n==1:
        # print(dataset_array_t.shape)
            for i in tqdm(range(len(dataset_array))):
                word_indices = np.nonzero(dataset_array[i])[0]
                dataset_words_new.append(word_indices.tolist())
            return dataset_words_new
        query_number = keyword_size
        padded_query_d = np.zeros((query_number,0))
        for i in tqdm(range(query_number)):
            s = np.count_nonzero(dataset_array_t[i])
            power = 0
            while(1):
                if s > (n*power):
                    power += 1
                else:
                    padding_number = (n*power)-s
                    break
            # print(padding_number)
            if padding_number > len(padded_query_d[0]):
                pad_doc = np.zeros((query_number, padding_number - len(padded_query_d[0])))
                padded_query_d = np.hstack((padded_query_d,pad_doc))
                temp = np.ones((1,padding_number))
                padded_query_d[i] = temp
            else:
                temp = np.zeros(len(padded_query_d[0]))
                index = np.where(temp==0)[0]

                pad_index = np.random.choice(index,padding_number,replace=False)
                temp[pad_index] = 1
                padded_query_d[i] = padded_query_d[i] + temp

        padded_query_d = np.hstack((dataset_array_t,padded_query_d))
        padded_query_d = padded_query_d.T
        for i in tqdm(range(len(padded_query_d))):
            word_indices = np.nonzero(padded_query_d[i])[0]
            dataset_words_new.append(word_indices.tolist())
        return dataset_words_new
    
    elif countermeasure == 'padding_cluster':
        print("Countermeasure is ",countermeasure)
        knum_in_cluster = 8
        if knum_in_cluster == 0 or knum_in_cluster == 1:
            for i in tqdm(range(len(dataset_array))):
                word_indices = np.nonzero(dataset_array[i])[0]
                dataset_words_new.append(word_indices.tolist())
            return dataset_words_new
        # v.shape should be (3000,1) 
        v = np.sum(dataset_array_t,axis=1)
        query_number = keyword_size
        index = [i for i in range(query_number)]
        id_v = list(zip(index,v))
        id_v = sorted(id_v,key=lambda x:x[1])
        i = 0
        padding_number = np.zeros((query_number,1))
        while i<query_number:
            if i < (query_number//knum_in_cluster)*knum_in_cluster:
                padding_number[i] = id_v[i-i%knum_in_cluster+knum_in_cluster-1][1] - id_v[i][1]
            else:
                padding_number[i] = id_v[-1][1] - id_v[i][1]
            i = i+1
        padded_query_d = np.zeros((query_number,0))
        for i in range(query_number):
            id = id_v[i][0]
            number = int(padding_number[i])
            if number > len(padded_query_d[0]):
                pad_doc = np.zeros((query_number, number - len(padded_query_d[0])))
                padded_query_d = np.hstack((padded_query_d,pad_doc))
                temp = np.ones((1,number))
                padded_query_d[id] = temp
            else:
                temp = np.zeros(len(padded_query_d[0]))
                index = np.where(temp==0)[0]
                pad_index = np.random.choice(index,number,replace=False)
                temp[pad_index] = 1
                padded_query_d[id] = padded_query_d[id] + temp
        padded_query_d = np.hstack((dataset_array_t,padded_query_d))
        padded_query_d = padded_query_d.T
        for i in tqdm(range(len(padded_query_d))):
            word_indices = np.nonzero(padded_query_d[i])[0]
            dataset_words_new.append(word_indices.tolist())
        return dataset_words_new
    
    else:
        return dataset_words


def process_once_wiki(dataset_name):
    # if bow file exists, skip
    if os.path.exists(os.path.join(RAW_DATA_PATH[dataset_name], f'bow-{dataset_name}.pkl')):
        print(f"Bow file for {dataset_name} already exists, skipping processing.")
        return
    
    if dataset_name == 'wiki_5000':
        raw_path = os.path.join(RAW_DATA_PATH[dataset_name],'kws_list_and_doc_kws_new_5000_0.pkl')
    elif dataset_name == 'wiki_7000':
        raw_path = os.path.join(RAW_DATA_PATH[dataset_name],'kws_list_and_doc_kws_new_7000_0.pkl')
    elif dataset_name == 'wiki_3000':
        raw_path = os.path.join(RAW_DATA_PATH[dataset_name],'kws_list_and_doc_kws_all_new_0.pkl')
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    mid_path = RAW_DATA_PATH[dataset_name]
    with open(raw_path, 'rb') as f:
        datas = pickle.load(f)

    datas_list = []
    for i in tqdm(range(len(datas[1]))):
        # get the index of no-zero elements
        index = np.nonzero(datas[1][i])[0]
        datas_list.append(index.tolist())

    # save datas_list to a file
    with open(os.path.join(mid_path, f'bow-{dataset_name}.pkl'), 'wb') as f:
        pickle.dump(datas_list, f)
        



def process_once_enron():
    raw_path = RAW_DATA_PATH['enron']
    mid_path = PRO_DATA_PATH['enron']
    keywords_read = read_keywords_from_file('enron')
    count = 0
    count_empty = 0
    count_invalid = 0
    messages = []
    messages_dates =[]
    for root, dirs, files in os.walk(raw_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='latin-1') as email_file:
                message = email.message_from_file(email_file)
            clearn_message = clearn_email(keywords_read, process_email(message))
            year_month = extract_date(message)
            year = year_month.split('-')[0]
            if len(clearn_message) <= 0:
                # print("Empty message.")
                count_empty += 1
                continue
            if year not in DESIRED_YEARS['enron']:
                # print("Invalid Year.")
                count_invalid += 1
                continue
            messages.append(clearn_message)
            messages_dates.append(year_month)
            count += 1
            if count % 10000 == 0:
                # print(messages[:5])
                print("{:d} documents processed".format(count))
    print("{:d} documents processed".format(count))
    print("{:d} documents empty".format(count_empty))
    print("{:d} documents invalid".format(count_invalid))
    with open(os.path.join(mid_path, 'enron_messages.pkl'), 'wb') as f:
        pickle.dump(messages, f)
    with open(os.path.join(mid_path, 'enron_messages_dates.pkl'), 'wb') as f:
        pickle.dump(messages_dates, f)
    print(messages_dates[:5])
    print('process raw enron finish √')




def divide_train_test(dataset_name, percentage=0.5, experiment_setting='sample',force_recompute=True,countermeasures=None,keyword_size=3000,adp=False):
    raw_path = RAW_DATA_PATH[dataset_name]
    mid_path = PRO_DATA_PATH[dataset_name]
    if os.path.exists(mid_path) and not force_recompute:
        print_str("Procession raw data", dataset_name)
    print_str("Processing raw data", dataset_name, mark_force=True)
    if experiment_setting == 'sample':
        output_path_train = os.path.join(PRO_DATA_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), 'train')
        output_path_test = os.path.join(PRO_DATA_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), 'test')
    else:
        output_path_train = os.path.join(PRO_DATA_PATH[dataset_name], str(percentage) + '-' + '1.0', 'train')
        output_path_test = os.path.join(PRO_DATA_PATH[dataset_name], str(percentage) + '-' + '1.0', 'test')
    #save to file
    os.makedirs(output_path_train, exist_ok=True)
    os.makedirs(output_path_test, exist_ok=True)
    dataset_words_train = []
    dataset_date_train = []
    dataset_index_train = []
    dataset_words_test = []
    dataset_date_test = []
    dataset_index_test = []
    keywords_read = read_keywords_from_file(dataset_name)
    if dataset_name == 'nytimes':
        # file_names, dates = date_loop(dataset_name, start_year=2001, end_year=2003)
        with open(raw_path, 'rb') as f:
            messages_keywords_dict = pickle.load(f)
        messages = messages_keywords_dict[0]
        random.shuffle(messages)
    
    elif dataset_name == 'wiki_3000' or dataset_name == 'wiki_5000' or dataset_name == 'wiki_7000':
        with open(os.path.join(raw_path, f'bow-{dataset_name}.pkl'), 'rb') as f:
            messages = pickle.load(f)
        random.shuffle(messages)
    

    elif dataset_name == 'enron':   
        with open(os.path.join(raw_path, 'enron_messages.pkl'), 'rb') as f:
            messages = pickle.load(f)
        with open(os.path.join(raw_path, 'enron_messages_dates.pkl'), 'rb') as f:
            message_dates = pickle.load(f)
        indexes_list = list(range(len(messages)))
        random.shuffle(indexes_list)
        messages_shuffled = [messages[i] for i in indexes_list]
        message_dates_shuffled = [message_dates[i] for i in indexes_list]
        messages = messages_shuffled
        message_dates = message_dates_shuffled
        
    index = list(range(len(messages)))
    index_mark = list(range(len(index)))
    random.shuffle(index_mark)
    split_point = int(len(index) * percentage)
    train_datas = [messages[idx] for idx in index_mark[:split_point]]
    train_index = [index[idx] for idx in index_mark[:split_point]]
    print(f'len(train_datas){len(train_datas)}', f'len(train_index){len(train_index)}')
    if experiment_setting == 'sample':
        test_datas = [messages[idx] for idx in index_mark[split_point:]]
        test_index = [index[idx] for idx in index_mark[split_point:]]
    else:
        test_datas = messages
        test_index = index
    # print(type(test_datas))
    print(f'len(test_datas){len(test_datas)}', f'len(test_index){len(test_index)}')
    # countermeasures to add here
    len_old_test_datas = len(test_datas)
    len_old_train_datas = len(train_datas)
    test_datas = apply_countermeasures(test_datas, countermeasures, keyword_size)
    # print(type(test_datas), f'len(train_index){len(train_index)}', f'len(old_train){len_old_test_datas}', f'len(new_train){len(train_datas)}')
    if adp == True:
        train_datas = apply_countermeasures(train_datas, countermeasures, keyword_size)
    file_names, dates = date_loop(dataset_name, start_year=2000, end_year=2002)
    # print(f'data:{dates}')
    for index in tqdm(range(len(train_datas))):
        message = train_datas[index]
        if dataset_name == 'nytimes' or dataset_name == 'wiki_3000' or dataset_name == 'wiki_5000' or dataset_name == 'wiki_7000' or (dataset_name == 'enron' and index >= len_old_train_datas):
            date_index = index % len(dates)
            date = dates[date_index]
        else :
            date = message_dates[train_index[index]]
        dataset_words_train.append(message)
        dataset_date_train.append(date)
        dataset_index_train.append(index)
    save_to_file(dataset_words_train, dataset_date_train, dataset_index_train, output_path_train)
    print(colors['cyan'].format(f'{dataset_name} train Dataset Save √'))
    for index in tqdm(range(len(test_datas))):
        message = test_datas[index]
        if dataset_name == 'nytimes' or dataset_name == 'wiki_3000' or dataset_name == 'wiki_5000' or dataset_name == 'wiki_7000' or (dataset_name == 'enron' and index >= len_old_test_datas):
            date_index = index % len(dates)
            date = dates[date_index]
        else :
            date = message_dates[test_index[index]]
        dataset_words_test.append(message)
        dataset_date_test.append(date)
        dataset_index_test.append(index)
    save_to_file(dataset_words_test, dataset_date_test, dataset_index_test, output_path_test)
    print(colors['cyan'].format(f'{dataset_name} test Dataset Save √'))




def save_email_by_month(dataset_name, force_save=False,experiment_setting='sample',part='train',percentage=0.5):
    if experiment_setting == 'sample':
        input_path = os.path.join(PRO_DATA_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part)
        output_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part)
    else:
        input_path = os.path.join(PRO_DATA_PATH[dataset_name], str(percentage) + '-' + '1.0', part)
        output_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + '1.0', part)
    if not force_save:
        print_str("Monthly Save email", dataset_name)
        return
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()
    print_str("Monthly Save email", dataset_name, mark_force=True)
    dataset_words, dataset_date, dataset_index = read_from_file(input_path)
    data_by_month = defaultdict(list)
    index_by_month = defaultdict(list)
    for message, date, index in zip(dataset_words, dataset_date, dataset_index):
        data_by_month[date].append(message)
        index_by_month[date].append(index)
    dates = []
    all_messages_list = []
    if dataset_name == "enron":
        dates = [
            '2000-01', '2000-02', '2000-03', '2000-04', '2000-05', '2000-06',
            '2000-07', '2000-08', '2000-09', '2000-10', '2000-11', '2000-12',
            '2001-01', '2001-02', '2001-03', '2001-04', '2001-05', '2001-06',
            '2001-07', '2001-08', '2001-09', '2001-10', '2001-11', '2001-12',
            '2002-01', '2002-02', '2002-03', '2002-04', '2002-05', '2002-06',
            '2002-07'
        ]
        for date in dates:
            message = data_by_month[date]
            index = index_by_month[date]
            all_messages_list += message
            os.makedirs(output_path, exist_ok=True)
            with open(os.path.join(output_path, f'{date}.pkl'), 'wb') as f:
                pickle.dump(message, f)
                pickle.dump(index, f)
    elif dataset_name == "lucene" or dataset_name == "nytimes" or dataset_name == "wiki_3000" or dataset_name == "wiki_5000" or dataset_name == "wiki_7000":
        dates = []
        for date in data_by_month.keys():
            message = data_by_month[date]
            index = index_by_month[date]
            dates.append(date)
            all_messages_list += message
            with open(os.path.join(output_path, f'{date}.pkl'), 'wb') as f:
                pickle.dump(message, f)
                pickle.dump(index, f)
    sorted_dates = sorted(dates, key=lambda x: datetime.strptime(x, "%Y-%m"))
    with open(os.path.join(output_path, 'dates.pkl'), 'wb') as f:
        pickle.dump(sorted_dates, f)

def get_fpbp(dataset_name, percentage=0.50, experiment_setting='sample', part='test', del_percentage=0.05):
    if experiment_setting == 'sample':
        input_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part)
        output_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part)
    else:
        input_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + '1.0', part)
        output_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + '1.0', part)
    print('fpbp del percentage', del_percentage)
    dates = get_dates(dataset_name)
    for date in dates:
        with open(os.path.join(input_path, f'{date}.pkl'), 'rb') as f:
            message = pickle.load(f)
            index = pickle.load(f)
        # print(f'len(message){len(message)}', f'len(index){len(index)}')
        messages_with_index = list(zip(message, index))
        random.shuffle(messages_with_index)
        
        split_point = int(len(messages_with_index) * del_percentage)
        
        remaining_messages = []
        remaining_indices = []
        
        for msg, idx in messages_with_index[split_point:]:
            remaining_messages.append(msg)
            remaining_indices.append(idx)
            
        with open(os.path.join(output_path, f'{date}.pkl'), 'wb') as f:
            pickle.dump(remaining_messages, f)
            pickle.dump(remaining_indices, f)


def accumulation_every_month_for_raw_dataset(dataset_name, percentage=0.50, part='train', force_division=False,experiment_setting='sample'):
    if experiment_setting == 'sample':
        input_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part)
        output_path = os.path.join(ACCUMULATION_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part)
    else:
        input_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + '1.0', part)
        output_path = os.path.join(ACCUMULATION_PATH[dataset_name], str(percentage) + '-' + '1.0', part)
    if os.path.exists(output_path) and not force_division:
        print_str("Monthly accumulate email", dataset_name, part=part, mark_part=True)
        return
    print_str("Monthly accumulate email", dataset_name, part=part, mark_part=True, mark_force=True)
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)
    dates = get_dates(dataset_name)
    acc_message = []
    acc_index = []
    for date in dates:
        # print(f'processing {date} dataset')
        with open(os.path.join(input_path, f'{date}.pkl'), 'rb') as f:
            message = pickle.load(f)
            index = pickle.load(f)
        acc_message = acc_message + message
        acc_index = acc_index + index
        # print(len(acc_index))
        with open(os.path.join(output_path, f'{date}.pkl'), 'wb') as f:
            pickle.dump(acc_message, f)
            pickle.dump(acc_index, f)
    end_time = time.time()
    print(str(end_time - start_time) + " seconds elapsed")


def save_division_by_dict(dataset_name, percentage=0.50, part='train', force_save_dict=False,experiment_setting='sample',countermeasures=None):
    if experiment_setting == 'sample':
        input_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part)
        transfer_path = os.path.join(TRANSFER_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part,countermeasures)
        output_path = os.path.join(DIVISION_PROCESSED_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part,countermeasures)
    else:   
        input_path = os.path.join(DIVISION_PATH[dataset_name], str(percentage) + '-' + '1.0', part)
        transfer_path = os.path.join(TRANSFER_PATH[dataset_name], str(percentage) + '-' + '1.0', part,countermeasures)
        output_path = os.path.join(DIVISION_PROCESSED_PATH[dataset_name], str(percentage) + '-' + '1.0', part,countermeasures)
    if os.path.exists(output_path) and not force_save_dict:
        print_str("Monthly save divisional dict", dataset_name, part=part, mark_part=True)
        return
    print_str("Monthly save divisional dict", dataset_name, part=part, mark_part=True, mark_force=True)
    start_time = time.time()
    os.makedirs(transfer_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    dates = get_dates(dataset_name)
    keywords_read = read_keywords_from_file(dataset_name)
    for date in dates:
        # print('processing {:s} dataset'.format(colors['cyan'].format(date)))
        keywords_transfer = {key: [] for key in keywords_read}
        keywords_list = list(keywords_read.keys())
        with open(os.path.join(input_path, f'{date}.pkl'), 'rb') as f:
            messages = pickle.load(f)
            indexes = pickle.load(f)
        for message, index in zip(messages, indexes):
            for word_index in message:
                word = keywords_list[word_index]
                keywords_transfer[word].append(index)
        with open(os.path.join(transfer_path, f'{date}.pkl'), 'wb') as f:
            pickle.dump(keywords_transfer, f)
        keywords_month = {}
        for word, index_set in keywords_transfer.items():
            dictionary = {'size': len(index_set), 'length': index_set}
            keywords_month[word] = dictionary
        date = format_date(date)
        with open(os.path.join(output_path, f'{date}.pkl'), 'wb') as f:
            pickle.dump(keywords_month, f)
    end_time = time.time()
    print(str(end_time - start_time) + " seconds elapsed")


def save_accumulation_by_dict(dataset_name, percentage=0.50, part='train', force_save_dict=False,experiment_setting='sample',countermeasures=None):
    if experiment_setting == 'sample':
        transfer_path = os.path.join(TRANSFER_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part,countermeasures )
        output_path = os.path.join(ACCUMULATION_PROCESSED_PATH[dataset_name], str(percentage) + '-' + str(1 - percentage), part,countermeasures)
    else:
        transfer_path = os.path.join(TRANSFER_PATH[dataset_name], str(percentage) + '-' + '1.0', part,countermeasures)
        output_path = os.path.join(ACCUMULATION_PROCESSED_PATH[dataset_name], str(percentage) + '-' + '1.0', part,countermeasures)
    if os.path.exists(output_path) and not force_save_dict:
        print_str("Monthly save accumulated dict", dataset_name, part=part, mark_part=True)
        return
    print_str("Monthly save accumulated dict", dataset_name, part=part, mark_part=True, mark_force=True)
    start_time = time.time()
    os.makedirs(output_path, exist_ok=True)
    dates = get_dates(dataset_name)
    keywords_read = read_keywords_from_file(dataset_name)
    keywords_tool = {key: [] for key in keywords_read}
    keywords_acc = {}
    for date in dates:
        # print('processing {:s} dataset'.format(colors['cyan'].format(date)))
        with open(os.path.join(transfer_path, f'{date}.pkl'), 'rb') as f:
            keywords_transfer = pickle.load(f)
        for word in keywords_tool.keys():
            keywords_tool[word] += keywords_transfer[word]
            dictionary = {'size': len(keywords_tool[word]), 'length': keywords_tool[word]}
            keywords_acc[word] = dictionary
        date = format_date(date)
        with open(os.path.join(output_path, f'{date}.pkl'), 'wb') as f:
            pickle.dump(keywords_acc, f)
    end_time = time.time()
    print(str(end_time - start_time) + " seconds elapsed")


if __name__ == '__main__':
    # process_once_enron()
    # process_once_wiki()
    parser = argparse.ArgumentParser(description="Process dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The dataset name",
        required=True,
    )
    parser.add_argument(
        "--experiment_setting",
        type=str,
        help="The experiment setting",
        required=True,
    )
    parser.add_argument(
        "--countermeasures",
        type=str,
        help="The countermeasures",
        required=True,
    )
    parser.add_argument(
        "--keyword_size",
        type=int,
        help="The keyword size",
        required=True,
    )
    parser.add_argument(
        "--adp",
        type=str,
        help="The adp",
        required=True,
        default=False,
    )
    parser.add_argument(
        "--force_recompute",
        type=bool,
        help="The force recompute",
        required=True,
    )
    parser.add_argument(
        "--start_percentage",
        type=int,
        help="The start percentage",
        required=True,
    )
    parser.add_argument(
        "--end_percentage",
        type=int,
        help="The end percentage",
        required=True,
    )
    parser.add_argument(
        "--step_percentage",
        type=int,
        help="The step percentage",
        required=True,
    )
    parser.add_argument(
        "--fpbp_del_percentage",
        type=float,
        help="The fpbp del percentage",
        required=True,
    )
    args = parser.parse_args()

    dataset_names = [args.dataset_name]
    experiment_setting = args.experiment_setting
    countermeasures = [args.countermeasures]
    keyword_size = args.keyword_size
    if args.adp == "true":  
        adp = True
    else:
        adp = False
    force_recompute = args.force_recompute
    start_percentage = args.start_percentage
    end_percentage = args.end_percentage
    step_percentage = args.step_percentage
    fpbp_del_percentage = args.fpbp_del_percentage
    percentage_list = list(range(start_percentage, (end_percentage + step_percentage), step_percentage))
    print(percentage_list)
    percentage_list = [percentage / 100 for percentage in percentage_list]

    print("adaptive strategy is", adp)

    for dataset_name in dataset_names:             
        print(colors['red'].format(dataset_name))  
        for percentage in percentage_list:
            print(f"For {colors['yellow'].format(percentage)} percentage")
            for countermeasure in countermeasures:
                divide_train_test(dataset_name, percentage=percentage, experiment_setting= experiment_setting,countermeasures=countermeasure,keyword_size=keyword_size,adp=adp)
                save_email_by_month(dataset_name, force_save=True,experiment_setting=experiment_setting,part='train',percentage=percentage)
                save_email_by_month(dataset_name, force_save=True,experiment_setting=experiment_setting,part='test',percentage=percentage)
                get_fpbp(dataset_name, percentage=percentage, experiment_setting=experiment_setting, part='test', del_percentage=fpbp_del_percentage)  
                accumulation_every_month_for_raw_dataset(dataset_name, percentage=percentage, part='train', force_division=True,experiment_setting=experiment_setting)
                accumulation_every_month_for_raw_dataset(dataset_name, percentage=percentage, part='test', force_division=True,experiment_setting=experiment_setting)
                save_division_by_dict(dataset_name, percentage=percentage, part='train', force_save_dict=True,experiment_setting=experiment_setting,countermeasures=countermeasure)
                save_division_by_dict(dataset_name, percentage=percentage, part='test', force_save_dict=True,experiment_setting=experiment_setting,countermeasures=countermeasure)
                save_accumulation_by_dict(dataset_name, percentage=percentage, part='train', force_save_dict=True,experiment_setting=experiment_setting,countermeasures=countermeasure)
                save_accumulation_by_dict(dataset_name, percentage=percentage, part='test', force_save_dict=True,experiment_setting=experiment_setting,countermeasures=countermeasure)
            print(f"{colors['yellow'].format(percentage)} percentage Finish √")
        print(f"{colors['red'].format(dataset_name)} Finish √")
    print("All finished √")