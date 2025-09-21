import pickle
import time
from tqdm import tqdm
from cal_acc import calculate_acc_weighted
from run_single_attack import *
import os
import argparse
def run_Jigsaw_IHOP_and_RSA_against_countermeasure(countermeasure,test_times=1,kws_uni_size=1000,\
                                                 datasets=["enron"],kws_extraction="sorted",observe_query_number_per_week = 500,\
                                                observe_weeks = 50,time_offset = 0,refspeed=2000,beta=0.9):
    if not os.path.exists("./result"):
        os.makedirs("./result")
    if not os.path.exists("./result/test_against_countermeasures"):
        os.makedirs("./result/test_against_countermeasures")
    print("Test Jigsaw IHOP and RSA against countermeasure")
    for dataset in datasets:
        if countermeasure =="padding_linear":
            if dataset == "wiki":
                Countermeasure_params = [
                {"alg":"padding_linear_2","n":0},
                {"alg":"padding_linear_2","n":50000},
                {"alg":"padding_linear_2","n":100000},
                {"alg":"padding_linear_2","n":150000}]
            else:
                Countermeasure_params = [\
                {"alg":"padding_linear_2","n":500}]
        elif countermeasure == "obfuscation":
            if dataset == "wiki":
                Countermeasure_params=[{"alg":"obfuscation","p":1,"q":0,"m":1},\
                {"alg":"obfuscation","p":0.999,"q":0.1,"m":1},\
                {"alg":"obfuscation","p":0.999,"q":0.2,"m":1},\
                {"alg":"obfuscation","p":0.999,"q":0.3,"m":1}
                ]
            else:
                Countermeasure_params=[{"alg":"obfuscation","p":1,"q":0,"m":1},\
                    {"alg":"obfuscation","p":0.999,"q":0.01,"m":1},\
                    {"alg":"obfuscation","p":0.999,"q":0.02,"m":1},\
                    {"alg":"obfuscation","p":0.999,"q":0.05,"m":1}
                    ]
        elif countermeasure == "padding_cluster":
            Countermeasure_params = [
                {"alg":"padding_cluster","knum_in_cluster":8}]
        elif countermeasure == "padding_seal":
            Countermeasure_params = [
                {"alg":"padding_seal","n":1},
                {"alg":"padding_seal","n":2},
                # {"alg":"padding_seal","n":3},
                # {"alg":"padding_seal","n":4},
                
                ]
        else:
            print("Countermeasure not supported")
            return
        for countermeasure_params in Countermeasure_params:
            Jigsaw_Result = []
            IHOP_Result = []
            RSA_Result = []
            Jigsaw_acc = []
            for i in tqdm(range(test_times)):
                rsa_attack_params={
                    "alg": "RSA",
                    "refinespeed":2070,
                    "known_query_number":5
                }
                ihop_attack_params={
                    "alg":"IHOP",
                    "niters":3,
                    "pfree":0.25,
                    "no_F":True
                    }
                our_attack_params={
                    "alg": "Ours",
                    "refinespeed":1800,
                    "alpha":0.3,
                    "beta":0.9,
                    "baseRec":45,
                    "confRec":35,
                    "step":3,
                    "no_F":True
                }
                if dataset == "wiki":
                    our_attack_params["refinespeed_exp"] = True
                    rsa_attack_params["refinespeed_exp"] = True
                else:
                    our_attack_params["refinespeed_exp"] = False
                    rsa_attack_params["refinespeed_exp"] = False

##################Jigsaw###################
                print(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                countermeasure_params,our_attack_params)
                result = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                countermeasure_params,our_attack_params)
                
                data_for_acc_cal = result["data_for_acc_cal"]

                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][0])
                # print({"Jigsaw step1:  dataset":dataset,"countermeasure_params":countermeasure_params,"acc":acc})

                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][1])
                # print({"Jigsaw step2:  dataset":dataset,"countermeasure_params":countermeasure_params,"acc":acc})


                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][2])
                print({"Jigsaw:  dataset":dataset,"countermeasure_params":countermeasure_params,"acc":acc})
                Jigsaw_Result.append((dataset,countermeasure_params,acc,result))
                Jigsaw_acc.append(acc)
                
 ################RSA##############

                result = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                countermeasure_params,rsa_attack_params)
                data_for_acc_cal = result["data_for_acc_cal"]
                for key in result["results"][1].keys():
                    del result["results"][0][key]
                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"][0])
                print({"RSA:   dataset":dataset,"countermeasure_params":countermeasure_params,"acc":acc})
                RSA_Result.append((dataset,countermeasure_params,acc,result))

################IHOP#################################
                result = run_single_attack(kws_uni_size,kws_uni_size,kws_extraction,observe_query_number_per_week,\
                    observe_weeks,time_offset,dataset,
                countermeasure_params,ihop_attack_params)
                data_for_acc_cal = result["data_for_acc_cal"]
                correct_count,acc,correct_id,wrong_id = \
                    calculate_acc_weighted(data_for_acc_cal,result["results"])
                print({"IHOP:   dataset":dataset,"countermeasure_params":countermeasure_params,"acc":acc})
                IHOP_Result.append((dataset,countermeasure_params,acc,result))

            if countermeasure_params["alg"] == "padding_linear_2":
                with open("./result/test_against_countermeasures/Jigsaw_"+dataset+\
                    "_padding_linear_n_"+str(countermeasure_params["n"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(Jigsaw_Result,f)
                with open("./result/test_against_countermeasures/RSA_"+dataset+\
                        "_padding_linear_n_"+str(countermeasure_params["n"])+\
                        "_kws_uni_size_"+str(kws_uni_size)+\
                        "_test_times_"+str(test_times)+".pkl", "wb") as f:
                        pickle.dump(RSA_Result,f)
                with open("./result/test_against_countermeasures/IHOP_"+dataset+\
                    "_padding_linear_n_"+str(countermeasure_params["n"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(IHOP_Result,f)
            elif countermeasure_params["alg"] == "obfuscation":
                with open("./result/test_against_countermeasures/Jigsaw_"+dataset+\
                    "_obfuscation_q_"+str(countermeasure_params["q"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(Jigsaw_Result,f)
                with open("./result/test_against_countermeasures/RSA_"+dataset+\
                    "_obfuscation_q_"+str(countermeasure_params["q"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(RSA_Result,f)
                with open("./result/test_against_countermeasures/IHOP_"+dataset+\
                    "_obfuscation_q_"+str(countermeasure_params["q"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(IHOP_Result,f)
            elif countermeasure_params["alg"] == "padding_cluster":
                with open("./result/test_against_countermeasures/Jigsaw_"+dataset+\
                    "_padding_cluster_knum_in_cluster_"+str(countermeasure_params["knum_in_cluster"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(Jigsaw_Result,f)
                with open("./result/test_against_countermeasures/RSA_"+dataset+\
                    "_padding_cluster_knum_in_cluster_"+str(countermeasure_params["knum_in_cluster"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(RSA_Result,f)
                with open("./result/test_against_countermeasures/IHOP_"+dataset+\
                    "_padding_cluster_knum_in_cluster_"+str(countermeasure_params["knum_in_cluster"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(IHOP_Result,f)
            elif countermeasure_params["alg"] == "padding_seal":
                with open("./result/test_against_countermeasures/Jigsaw_"+dataset+\
                    "_padding_seal_"+str(countermeasure_params["n"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl", "wb") as f:
                    pickle.dump(Jigsaw_Result,f)
                with open("./result/test_against_countermeasures/RSA_"+dataset+\
                        "_padding_seal_"+str(countermeasure_params["n"])+\
                        "_kws_uni_size_"+str(kws_uni_size)+\
                        "_test_times_"+str(test_times)+".pkl", "wb") as f:
                        pickle.dump(RSA_Result,f)
                with open("./result/test_against_countermeasures/IHOP_"+dataset+\
                    "_padding_seal_"+str(countermeasure_params["n"])+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
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
    
    run_Jigsaw_IHOP_and_RSA_against_countermeasure(
        args.countermeasure,
        test_times=args.test_times,
        kws_uni_size=args.kws_uni_size,
        datasets=args.datasets,
        kws_extraction=args.kws_extraction
    )
