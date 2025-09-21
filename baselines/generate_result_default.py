import json
import argparse
def calc_median(arr):
    n = len(arr)
    if n % 2 == 0:
        return (arr[n//2 - 1] + arr[n//2]) / 2
    else:
        return arr[n//2]

def generate_result(dataset,kws_uni_size):
    with open(f"result/test_attacks_default/Re_{dataset}_{kws_uni_size}.json","r") as f:
        data = json.load(f)

        print(data)
        acc_keys = ["Jigsaw_acc","RSA_acc","IHOP_acc"]
        time_keys = ["Jigsaw_time","RSA_time","IHOP_time"]
        # like above, calculate the min, max, median, Q1, Q3, IQR, lower_fence, upper_fence
        for key in acc_keys:
            # j is the index of key in acc_keys
            j = acc_keys.index(key)
            print(key)
            acc_data = data[key][0]
            # print(acc_data)
            # data_acc = data["Our_acc"]

            accuracies = sorted(acc_data)
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

            method_name = key.replace("_acc", "")

            print(f"\\addplot+ [fill=RYB2, draw=RYB2, boxplot prepared={{draw position=1.5,")
            print(f"lower whisker={min_value}, lower quartile={Q1},")
            print(f"median={median}, upper quartile={Q3},")
            print(f"upper whisker={max_value}, sample size={len(accuracies)},}},")
            print(f"] coordinates {{}};")

            average_runtime = data[time_keys[j]][0]

            print(f"\n\\addplot[only marks, mark=star, color=black] coordinates {{")
            print(f"    (1.5, {average_runtime})")
            print(f"}};")

            with open(f"final_result/test_attacks_default/{method_name}_{dataset}_{kws_uni_size}.txt", 'w') as f:
                f.write(f"min_value: {min_value}\n")
                f.write(f"Q1: {Q1}\n")
                f.write(f"median: {median}\n")
                f.write(f"Q3: {Q3}\n")
                f.write(f"max_value: {max_value}\n")
                f.write(f"average_runtime: {average_runtime}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="enron")
    parser.add_argument("--kws_uni_size", type=int, default=3000)
    args = parser.parse_args()
    generate_result(args.dataset, args.kws_uni_size)
