import pickle
from matplotlib import pyplot as plt
import numpy as np
import json
import argparse
import os
def generate_against_countermeasures(dataset,countermeasure,kws_uni_size,test_times=30,legend=True):
    N = [0,500,1000,1500]
    Jigsaw_result = []
    RSA_result = []
    IHOP_result = []
    IHOP_alpha_result = []
    if countermeasure == "padding_linear":
        if dataset== "wiki":
            params = [0,50000,100000,150000]
        else:
            params = [500]
        string = "_padding_linear_n_"
    elif countermeasure == "padding_cluster":
        params = [8]
        string = "_padding_cluster_knum_in_cluster_"
    elif  countermeasure == "padding_seal":
        params = [2]
        string = "_padding_seal_"
    for param in params:
        with open("result/test_against_countermeasures/Jigsaw_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            Jigsaw_result.append(result)
        with open("result/test_against_countermeasures/RSA_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            RSA_result.append(result)
        with open("result/test_against_countermeasures/IHOP_"+dataset+\
                string+str(param)+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
            result = pickle.load(f)
            IHOP_result.append(result)
        
        
    
    Jigsaw_acc = []
    RSA_acc = []
    IHOP_acc = []
    # IHOP_alpha_acc = []
    Jigsaw_time = []
    RSA_time = []
    IHOP_time = []
    

    
    for result in Jigsaw_result:
        com_overhead = []
        sto_overhead = []
        acc = []
        attack_time = []
        for r in result:
            acc.append(r[2])
            attack_time.append(r[3]["Attack_time"])
            com_overhead.append(r[3]["data_for_acc_cal"]["communication overhead"])
            sto_overhead.append(r[3]["data_for_acc_cal"]["storage overhead"])
        Jigsaw_acc.append(acc)
        Jigsaw_time.append(np.average(attack_time))
        print("Communication overhead:",np.average(com_overhead))
        print("Storage overhead:",np.average(sto_overhead))
    for result in RSA_result:
        acc = []
        attack_time = []
        for r in result:
            acc.append(r[2])
            attack_time.append(r[3]["Attack_time"])
        RSA_acc.append(acc)
        RSA_time.append(np.average(attack_time))
    for result in IHOP_result:
        acc = []
        attack_time = []
        for r in result:
            acc.append(r[2])
            attack_time.append(r[3]["Attack_time"])
        IHOP_acc.append(acc)
        IHOP_time.append(np.average(attack_time))
    
    # save the results in a json file
    final_dir = "result/test_against_countermeasures/Re_"+dataset+"_"+countermeasure+"_"+str(kws_uni_size)+".json"
    # if dir didn't exist, create it
    if not os.path.exists(os.path.dirname(final_dir)):
        os.makedirs(os.path.dirname(final_dir))
    with open(final_dir,"w") as f:
        json.dump({"Jigsaw_acc":Jigsaw_acc,"Jigsaw_time":Jigsaw_time,"RSA_acc":RSA_acc,"RSA_time":RSA_time,"IHOP_acc":IHOP_acc,"IHOP_time":IHOP_time},f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--countermeasure", type=str, default="padding_linear_2")
    parser.add_argument("--test_times", type=int, default=30)
    parser.add_argument("--kws_uni_size", type=int, default=3000)
    parser.add_argument("--datasets", type=str, default="enron")
    parser.add_argument("--kws_extraction", type=str, default="sorted")
    args = parser.parse_args()
    print(f"Datasets type: {type(args.datasets)}, value: {args.datasets}")
    generate_against_countermeasures(args.datasets,args.countermeasure,args.kws_uni_size,test_times=args.test_times)

   