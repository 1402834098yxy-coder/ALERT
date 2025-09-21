import os
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

RAW_DATA_PATH = {'enron': f'{BASE_PATH}/',
                 'nytimes': f'{BASE_PATH}/bow-nytimes.pkl',
                 'wiki_3000': f'{BASE_PATH}/',
                 'wiki_5000': f'{BASE_PATH}/',
                 'wiki_7000': f'{BASE_PATH}/'}
PRO_DATA_PATH = {'enron': f'{BASE_PATH}/abstract/enron/Pro_data',
                 'nytimes': f'{BASE_PATH}/abstract/nytimes/Pro_data/',
                 'wiki_3000': f'{BASE_PATH}/abstract/wiki_3000/Pro_data/',
                 'wiki_5000': f'{BASE_PATH}/abstract/wiki_5000/Pro_data/',
                 'wiki_7000': f'{BASE_PATH}/abstract/wiki_7000/Pro_data/'}
CLASSIFY_PATH = {'enron': f'{BASE_PATH}/abstract/enron/Classify_By_Month',
                 'nytimes': f'{BASE_PATH}/abstract/nytimes/Classify_By_Month/',
                 'wiki_3000': f'{BASE_PATH}/abstract/wiki_3000/Classify_By_Month/',
                 'wiki_5000': f'{BASE_PATH}/abstract/wiki_5000/Classify_By_Month/',
                 'wiki_7000': f'{BASE_PATH}/abstract/wiki_7000/Classify_By_Month/'}
DIVISION_PATH = {'enron': f'{BASE_PATH}/raw/enron/Division',
                 'nytimes': f'{BASE_PATH}/raw/nytimes/Division',
                 'wiki_3000': f'{BASE_PATH}/raw/wiki_3000/Division',
                 'wiki_5000': f'{BASE_PATH}/raw/wiki_5000/Division',
                 'wiki_7000': f'{BASE_PATH}/raw/wiki_7000/Division'}
TRANSFER_PATH = {'enron': f'{BASE_PATH}/transfer/enron',
                 'nytimes': f'{BASE_PATH}/transfer/nytimes',
                'wiki_3000': f'{BASE_PATH}/transfer/wiki_3000',
                 'wiki_5000': f'{BASE_PATH}/transfer/wiki_5000',
                 'wiki_7000': f'{BASE_PATH}/transfer/wiki_7000'}
EXTRACT_RAW_PATH = {'enron': f'{BASE_PATH}/extract/enron/Raw',
                    'nytimes': f'{BASE_PATH}/extract/nytimes/Raw',
                    'wiki_3000': f'{BASE_PATH}/extract/wiki_3000/Raw',
                    'wiki_5000': f'{BASE_PATH}/extract/wiki_5000/Raw',
                    'wiki_7000': f'{BASE_PATH}/extract/wiki_7000/Raw'}
EXTRACT_DIVISION_PATH = {'enron': f'{BASE_PATH}/extract/enron/Division',
                         'nytimes': f'{BASE_PATH}/extract/nytimes/Division',
                         'wiki_3000': f'{BASE_PATH}/extract/wiki_3000/Division',
                         'wiki_5000': f'{BASE_PATH}/extract/wiki_5000/Division',
                         'wiki_7000': f'{BASE_PATH}/extract/wiki_7000/Division'}
EXTRACT_ACCUMULATION_PATH = {'enron': f'{BASE_PATH}/extract/enron/Accumulation',
                             'nytimes': f'{BASE_PATH}/extract/nytimes/Accumulation',
                             'wiki_3000': f'{BASE_PATH}/extract/wiki_3000/Accumulation',
                             'wiki_5000': f'{BASE_PATH}/extract/wiki_5000/Accumulation',
                             'wiki_7000': f'{BASE_PATH}/extract/wiki_7000/Accumulation'}
DIVISION_PROCESSED_PATH = {'enron': f'{BASE_PATH}/processed/enron/Division',
                           'nytimes': f'{BASE_PATH}/processed/nytimes/Division',
                           'wiki_3000': f'{BASE_PATH}/processed/wiki_3000/Division',
                           'wiki_5000': f'{BASE_PATH}/processed/wiki_5000/Division',
                           'wiki_7000': f'{BASE_PATH}/processed/wiki_7000/Division'}
ACCUMULATION_PATH = {'enron': f'{BASE_PATH}/raw/enron/Accumulation',
                     'nytimes': f'{BASE_PATH}/raw/nytimes/Accumulation',
                     'wiki_3000': f'{BASE_PATH}/raw/wiki_3000/Accumulation',
                     'wiki_5000': f'{BASE_PATH}/raw/wiki_5000/Accumulation',
                     'wiki_7000': f'{BASE_PATH}/raw/wiki_7000/Accumulation'}
ACCUMULATION_PROCESSED_PATH = {'enron': f'{BASE_PATH}/processed/enron/Accumulation',
                                'nytimes': f'{BASE_PATH}/processed/nytimes/Accumulation',
                               'wiki_3000': f'{BASE_PATH}/processed/wiki_3000/Accumulation',
                               'wiki_5000': f'{BASE_PATH}/processed/wiki_5000/Accumulation',
                               'wiki_7000': f'{BASE_PATH}/processed/wiki_7000/Accumulation'}
KEYWORDS_PATH = {'enron': f'{BASE_PATH}/enron-full_stems_to_words.pkl',
                 'nytimes': f'{BASE_PATH}/bow-nytimes.pkl',
                 'wiki_3000': f'{BASE_PATH}/kws_dict_3000_sorted.pkl',
                 'wiki_5000': f'{BASE_PATH}/kws_dict_5000_sorted.pkl',
                 'wiki_7000': f'{BASE_PATH}/kws_dict_7000_sorted.pkl'}
DESIRED_YEARS = {'enron': ["2000", "2001", "2002"],
                 'nytimes': ["2001", "2002", "2003"]}
colors = {'black': "\033[30m{}\033[0m",'red': "\033[31m{}\033[0m", 'green': "\033[32m{}\033[0m", 'yellow': "\033[33m{}\033[0m",
          'blue': "\033[34m{}\033[0m", 'purple': "\033[35m{}\033[0m", 'cyan': "\033[36m{}\033[0m", 'white': "\033[37m{}\033[0m"}
Co_Occurences_PATH = {'enron': f'{BASE_PATH}/processed/enron/Co_Occurences',
                      'nytimes': f'{BASE_PATH}/processed/nytimes_data/Co_Occurrences',
                      'wiki_3000': f'{BASE_PATH}/processed/wiki_3000/Co_Occurrences'}