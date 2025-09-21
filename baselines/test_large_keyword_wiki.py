import subprocess
import re
import json
import time
from run_single_attack import *
import os
import argparse

def get_block_result(accuracies,runtimes):

    accuracies = sorted(accuracies)
    min_value = accuracies[0]
    max_value = accuracies[-1]

    n = len(accuracies)
    if n % 2 == 0:
        median = (accuracies[n//2 - 1] + accuracies[n//2]) / 2
    else:
        median = accuracies[n//2]

    def calc_median(arr):
        n = len(arr)
        if n % 2 == 0:
            return (arr[n//2 - 1] + arr[n//2]) / 2
        else:
            return arr[n//2]

    if n % 2 == 0:
        Q1 = calc_median(accuracies[:n//2])
        Q3 = calc_median(accuracies[n//2:])
    else:
        Q1 = calc_median(accuracies[:n//2])
        Q3 = calc_median(accuracies[n//2 + 1:])

    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    min_value = max(min_value, lower_fence)
    max_value = min(max_value, upper_fence)

    print(f"\\addplot+ [fill=RYB2, draw=RYB2, boxplot prepared={{draw position=1.5,")
    print(f"lower whisker={min_value}, lower quartile={Q1},")
    print(f"median={median}, upper quartile={Q3},")
    print(f"upper whisker={max_value}, sample size={len(accuracies)},}},")
    print(f"] coordinates {{}};")

    average_runtime = sum(runtimes) / len(runtimes)

    print(f"\n\\addplot[only marks, mark=star, color=black] coordinates {{")
    print(f"    (1.5, {average_runtime})")
    print(f"}});")
    return min_value, Q1, median, Q3, max_value, len(accuracies),average_runtime

def run_attack(kws_uni_size,refspeed,dataset = ["enron"],query_number_per_week=2000,weeks=50):
    attack_params_without_freq = {"alg":"Ours","alpha":0.3,"step":3,\
                        "baseRec":45,"confRec":35,\
                        "beta":0.9,"no_F":True,"refinespeed":refspeed}
    result = run_single_attack(kws_uni_size,kws_uni_size,"sorted",query_number_per_week,weeks,0,dataset,\
                        {"alg":None},attack_params_without_freq)
    return result

def parse_output(output):
    if output is None:
        return {'accuracy': None, 'runtime': None}
    
    output = str(output)
    accuracy_match = re.search(r'Accuracy: (\d+\.\d+)', output)
    runtime_match = re.search(r'Run time: (\d+\.\d+)', output)
    
    accuracy = float(accuracy_match.group(1)) if accuracy_match else None
    runtime = float(runtime_match.group(1)) if runtime_match else None
    
    return {'accuracy': accuracy, 'runtime': runtime}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wiki")
    parser.add_argument("--kws_uni_size", type=int, default=7000)
    parser.add_argument("--repeat_time", type=int, default=30)
    parser.add_argument("--refspeed", type=int, default=7000)
    args = parser.parse_args()


    print("dataset: ", args.dataset)
    print("kws_uni_size: ", args.kws_uni_size)
    print("repeat_time: ", args.repeat_time)
    print("refspeed: ", args.refspeed)
    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/test_large_keyword_wiki"):
        os.makedirs("./result/test_large_keyword_wiki")
    repeat_time = args.repeat_time
    # Run the attack 30 times
    results = []
    log_content = ""
    total_accuracy = 0
    total_runtime = 0
    # run_time_limit = 10001
    accuracies = []
    runtimes = []
    dataset = args.dataset
    kws_uni_size = args.kws_uni_size
    refspeed = args.refspeed
    # print("limited time: ", run_time_limit)
    for i in range(repeat_time):
        print(f"Running attack {i+1}/{repeat_time}")
        result = run_attack(kws_uni_size,refspeed,dataset)
        log_content += f"--- Run {i+1} ---\n\n\n"
        data_for_acc_cal = result["data_for_acc_cal"]
        tdid_2_kwid = result["results"][2]
        correct_count,acc,correct_id,wrong_id=\
            calculate_acc_weighted(data_for_acc_cal,tdid_2_kwid)
        log_content += f"Attack Parameters: {result['attack_params']}\n\n"
        runtime = result["Attack_time"]
        # accuracy = result['accuracy']
        # runtime = result['runtime']
        total_accuracy += acc
        total_runtime += runtime
        accuracies.append(acc)
        runtimes.append(runtime)
        print(f"Run {i+1}: Accuracy = {acc:.4f}, Runtime = {runtime:.4f} seconds")
        log_content += f"Run {i+1}: Accuracy = {acc:.4f}, Runtime = {runtime:.4f} seconds\n\n\n"
        # results.append(f"Accurcay:{acc:.4f},\n Runtime:{runtime:.4f}")
        results.append({
            "accuracy": acc,
            "runtime": runtime
        })    
        time.sleep(0.1)  # Add a small delay between runs

    # Calculate averages
    avg_accuracy = total_accuracy / repeat_time
    avg_runtime = total_runtime / repeat_time

    print(f"\nAverage Accuracy: {avg_accuracy:.4f}")
    print(f"Average Runtime: {avg_runtime:.4f} seconds")

    min_value, Q1, median, Q3, max_value, len_data,average_runtime = get_block_result(accuracies,runtimes)

    # Save log file
    with open(f'result/test_large_keyword_wiki/attack_log_{dataset}_{kws_uni_size}_{refspeed}.txt', 'w') as log_file:
        log_file.write(log_content)

    # Create JSON file
    json_data = {
        'runs': results,
        'average_accuracy': avg_accuracy,
        'average_runtime': avg_runtime
    }

    with open(f'result/test_large_keyword_wiki/attack_results_{dataset}_{kws_uni_size}_{refspeed}.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=2)

    output_dir = f'final_result/test_large_keyword_wiki/results_{dataset}_{kws_uni_size}.txt'
    # if dir didn't exist, create it
    if not os.path.exists(os.path.dirname(output_dir)):
        os.makedirs(os.path.dirname(output_dir))

    with open(output_dir, 'w') as txt_file:
        txt_file.write(f"Min Value: {min_value:.4f}\n")
        txt_file.write(f"Q1: {Q1:.4f}\n")
        txt_file.write(f"Median: {median:.4f}\n")
        txt_file.write(f"Q3: {Q3:.4f}\n")
        txt_file.write(f"Max Value: {max_value:.4f}\n")
        txt_file.write(f"Number of Data: {len_data}\n")
        txt_file.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
        txt_file.write(f"Average Runtime: {average_runtime:.4f} seconds\n")

    print(f"Attack runs completed. Results saved in {output_dir}")
