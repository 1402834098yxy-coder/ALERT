import json
import os 
import re
import argparse

def read_log_file(input_dir, runtime, part):
    final_round_pattern = r"final round\s+(\d+)\s*\nfinal Accuracy: (\d+\.\d+)%"


    # For matching the overall accuracy
    overall_pattern = r"The overall accuracy is\s+(\d+\.\d+)"

    # For matching the total time
    total_time_pattern = r"The total time is\s+(\d+\.\d+)"

    overall_accuracies = []
    total_times = []
    final_rounds = []
    final_accuracies = []
    for i in range(1,runtime+1):
        file_path = os.path.join(input_dir + str(i)+ '_' + f'{part}.log')
        with open(file_path, 'r') as f:
            log_data = f.read()
            temp_round = []
            temp_accuracy = []
            final_round_matches = re.findall(final_round_pattern, log_data)
            for round_match, accuracy_match in final_round_matches:
                temp_round.append(int(round_match))
                temp_accuracy.append(float(accuracy_match))
            final_rounds.append(temp_round)
            final_accuracies.append(temp_accuracy)

            overall_match = re.search(overall_pattern, log_data)
            if overall_match:
                overall_accuracies.append(float(overall_match.group(1)))
            else:
                print(f"No overall accuracy found ")

            total_time_match = re.search(total_time_pattern, log_data)
            if total_time_match:
                total_times.append(float(total_time_match.group(1)))
            else:
                print(f"No total time found ")
                # set runtime to 0
                total_times.append(0)
    return overall_accuracies, total_times, final_rounds, final_accuracies

def calc_cumulative_average_accuracy(final_accuracies, keyword_num, runtime):
    threshold = 100
    # get a list of average accuracy with final accuracies and runtime
    average_accuracies = []
    
    # Get number of positions in each run
    num_positions = len(final_accuracies[0])
    print(final_accuracies)
    # For each position across all runs
    for pos in range(num_positions):
        pos_sum = 0
        # Sum up values at this position from each run
        for run in range(runtime):
            pos_sum += final_accuracies[run][pos]
        # Calculate average for this position
        pos_avg = pos_sum / runtime
        average_accuracies.append(pos_avg)
    print("average_accuracies",average_accuracies)
    # Convert final accuracies to average accuracy for each threshold
    cumulative_accuracies = []

    iteration = keyword_num / threshold
    temp_sum = 0
    for j in range(int(iteration)):
        temp_sum += average_accuracies[j]
        temp_avg = temp_sum / ((j+1)*threshold)
        cumulative_accuracies.append(temp_avg)
    print("cumulative_accuracies",cumulative_accuracies)
        
    return cumulative_accuracies



def calc_median(arr):
    n = len(arr)
    if n % 2 == 0:
        return (arr[n//2 - 1] + arr[n//2]) / 2
    else:
        return arr[n//2]

def generate_final_result(accuracies, total_times,runtime):

    if runtime == 1:
        min_value = accuracies[0]
        max_value = accuracies[0]
        median = accuracies[0]
        Q1 = accuracies[0]
        Q3 = accuracies[0]
        average_runtime = total_times[0]
        average_accuracy = accuracies[0]
    
   
        
    else:   
        accuracies = sorted(accuracies)

        min_value = accuracies[0]
        max_value = accuracies[-1]

        n = len(accuracies)
        if n % 2 == 0:
            median = (accuracies[n//2 - 1] + accuracies[n//2]) / 2
        else:
            median = accuracies[n//2]



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

        average_accuracy = sum(accuracies)/len(accuracies)

        print(f"\\addplot+ [fill=RYB2, draw=RYB2, boxplot prepared={{draw position=1.5,")
        print(f"lower whisker={min_value}, lower quartile={Q1},")
        print(f"median={median}, upper quartile={Q3},")
        print(f"upper whisker={max_value}, sample size={len(accuracies)},}},")
        print(f"] coordinates {{}};")

        average_runtime = sum(total_times)/len(total_times)

        print(f"\n\\addplot[only marks, mark=star, color=black] coordinates {{")
        print(f"    (1.5, {average_runtime})")
        print(f"}};")

    return min_value, max_value, median, Q1, Q3, average_runtime, average_accuracy

def save_result(dir, min_value, max_value, median, Q1, Q3, average_runtime, average_accuracy,cumulative_accuracy):
    output_file = os.path.join(dir + 'final_result.txt')
    # os.makedirs(output_file, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(f"min_value: {min_value}\n")
        f.write(f"Q1: {Q1}\n")
        f.write(f"median: {median}\n")
        f.write(f"Q3: {Q3}\n")
        f.write(f"max_value: {max_value}\n")
        f.write(f"average_runtime: {average_runtime}\n")
        f.write(f"average_accuracy: {average_accuracy}\n")
        f.write(f"cumulative_accuracy: {cumulative_accuracy}\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_base_dir", type=str, required=True, 
                      help="Directory containing the result JSON files")
    parser.add_argument("--runtime", type=str, required=True)
    parser.add_argument("--part", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--candidate", type=str, required=False, default="0.5-0.5")
    parser.add_argument("--keyword_size", type=str, required=True, default="3000")
    args = parser.parse_args()

    runtime = int(args.runtime)
    part = args.part
    input_dir = args.log_base_dir
    output_dir = args.output_dir
    output_base_dir = '/'.join(output_dir.rstrip('/').split('/')[:-1])
    keyword_num = int(args.keyword_size)
    os.makedirs(output_base_dir, exist_ok=True)

    # os.makedirs(output_dir, exist_ok=True)
    overall_accuracies, total_times, final_rounds, final_accuracies = read_log_file(input_dir, runtime, part)
    # print(final_accuracies)
    min_value, max_value, median, Q1, Q3, average_runtime, average_accuracy = generate_final_result(overall_accuracies, total_times,runtime)
    cumulative_accuracy = calc_cumulative_average_accuracy(final_accuracies,keyword_num,runtime)
    save_result(output_dir, min_value, max_value, median, Q1, Q3, average_runtime, average_accuracy,cumulative_accuracy)
    # print(min_value, max_value, median, Q1, Q3, average_runtime)
