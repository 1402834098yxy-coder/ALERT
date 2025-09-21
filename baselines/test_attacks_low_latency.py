import pickle
import time
from tqdm import tqdm
from cal_acc import calculate_acc_weighted
from run_single_attack import *
import os
import argparse
def run_Jigsaw_IHOP_and_RSA_low_latency(countermeasure,test_times=1,kws_uni_size=1000,\
                                                 datasets=["enron"],kws_extraction="sorted",observe_query_number_per_week = 500,\
                                                observe_weeks = 50,time_offset = 0,refspeed=2000,beta=0.9):
    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/test_attacks_low_latency"):
        os.makedirs("./result/test_attacks_low_latency")
    print("Test Jigsaw IHOP and RSA under low latency")
    desired_times = [5,10,15]
    Jigsaw_refinespeed = [2100,1800,1500]
    RSA_refinespeed = [2150,2070,2000]
    IHOP_niters = [2,3,5]
    for dataset in datasets:
        for desired_time in desired_times:
            desired_time_index = desired_times.index(desired_time)
            Jigsaw_Result = []
            IHOP_Result = []
            RSA_Result = []
            Jigsaw_acc = []
            for i in tqdm(range(test_times)):
                rsa_attack_params={
                    "alg": "RSA",
                    "refinespeed":RSA_refinespeed[desired_time_index],
                    "known_query_number":5
                }
                ihop_attack_params={
                    "alg":"IHOP",
                    "niters":IHOP_niters[desired_time_index],
                    "pfree":0.25,
                    "no_F":True
                    }
                our_attack_params={
                    "alg": "Ours",
                    "refinespeed":Jigsaw_refinespeed[desired_time_index],
                    "alpha":0.3,
                    "beta":0.9,
                    "baseRec":45,
                    "confRec":35,
                    "step":3,
                    "no_F":True
                }
                print(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                {"alg":None},our_attack_params)
                result = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                {"alg":None},our_attack_params)
                
                data_for_acc_cal = result["data_for_acc_cal"]

                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][0])
                # print({"Jigsaw step1:  dataset":dataset,"countermeasure_params":countermeasure_params,"acc":acc})

                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][1])
                # print({"Jigsaw step2:  dataset":dataset,"countermeasure_params":countermeasure_params,"acc":acc})


                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][2])
                print({"Jigsaw:  dataset":dataset, "desired_time":desired_time, "refspeed":Jigsaw_refinespeed[desired_time_index],"acc":acc})
                Jigsaw_Result.append((dataset,Jigsaw_refinespeed[desired_time_index],acc,result))
                Jigsaw_acc.append(acc)
                
 ################RSA##############

                result = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                {"alg":None},rsa_attack_params)
                data_for_acc_cal = result["data_for_acc_cal"]
                for key in result["results"][1].keys():
                    del result["results"][0][key]
                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][0])
                print({"RSA:   dataset":dataset, "desired_time":desired_time, "refspeed":RSA_refinespeed[desired_time_index],"acc":acc})
                RSA_Result.append((dataset,RSA_refinespeed[desired_time_index],acc,result))

################IHOP#################################
                result = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                {"alg":None},ihop_attack_params)
                data_for_acc_cal = result["data_for_acc_cal"]
                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"])
                print({"IHOP:   dataset":dataset, "desired_time":desired_time, "niters":IHOP_niters[desired_time_index],"acc":acc})
                IHOP_Result.append((dataset,IHOP_niters[desired_time_index],acc,result))

            # if desired_time == 5:
            with open("./result/test_attacks_low_latency/Jigsaw_"+dataset+\
                "_kws_uni_size_"+str(kws_uni_size)+ "_desired_time_"+str(desired_time)+\
                "_test_times_"+str(test_times)+".pkl", "wb") as f:
                pickle.dump(Jigsaw_Result,f)
            with open("./result/test_attacks_low_latency/RSA_"+dataset+\
                "_kws_uni_size_"+str(kws_uni_size)+ "_desired_time_"+str(desired_time)+\
                "_test_times_"+str(test_times)+".pkl", "wb") as f:
                pickle.dump(RSA_Result,f)
            with open("./result/test_attacks_low_latency/IHOP_"+dataset+\
                "_kws_uni_size_"+str(kws_uni_size)+ "_desired_time_"+str(desired_time)+\
                "_test_times_"+str(test_times)+".pkl", "wb") as f:
                pickle.dump(IHOP_Result,f)

    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--countermeasure", type=str, default="padding_linear_2")
    parser.add_argument("--test_times", type=int, default=30)
    parser.add_argument("--kws_uni_size", type=int, default=3000)
    parser.add_argument("--datasets", nargs='+', type=str, default=["enron"])
    parser.add_argument("--kws_extraction", type=str, default="sorted")
    args = parser.parse_args()
    
    # print(f"Datasets type: {type(args.datasets)}, value: {args.datasets}")
    
    run_Jigsaw_IHOP_and_RSA_low_latency(
        args.countermeasure,
        test_times=args.test_times,
        kws_uni_size=args.kws_uni_size,
        datasets=args.datasets,
        kws_extraction=args.kws_extraction
    )
