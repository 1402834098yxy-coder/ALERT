import pickle
from matplotlib import pyplot as plt
import numpy as np
import json
import argparse
import os
def generate_attacks_default(dataset,kws_uni_size,test_times=30,legend=True):
    Jigsaw_result = []
    RSA_result = []
    IHOP_result = []
    with open("./result/test_attacks_default/Jigsaw_"+dataset+\
                    "_kws_uni_size_"+str(kws_uni_size)+\
                    "_test_times_"+str(test_times)+".pkl","rb") as f:
        result = pickle.load(f)
        Jigsaw_result.append(result)
    with open("./result/test_attacks_default/RSA_"+dataset+\
                "_kws_uni_size_"+str(kws_uni_size)+\
                "_test_times_"+str(test_times)+".pkl","rb") as f:
        result = pickle.load(f)
        RSA_result.append(result)
    with open("./result/test_attacks_default/IHOP_"+dataset+\
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
        acc = []
        attack_time = []
        for r in result:
            acc.append(r[2])
            attack_time.append(r[3]["Attack_time"])
        Jigsaw_acc.append(acc)
        Jigsaw_time.append(np.average(attack_time))
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
    final_dir = "result/test_attacks_default/Re_"+dataset+"_"+str(kws_uni_size)+".json"
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
    generate_attacks_default(args.datasets,args.kws_uni_size,test_times=args.test_times)

    # generate_against_countermeasures("enron","padding",3000,test_times=30,legend=True)
    # generate_against_countermeasures("nytimes","padding",3000,test_times=30,legend=False)
    # generate_against_countermeasures("enron","obfuscation",1000,test_times=30,legend=True)
    # generate_against_countermeasures("lucene","obfuscation",1000,test_times=30,legend=False)


    # generate_against_countermeasures("wiki","obfuscation",1000,test_times=10,legend=True)
    # generate_against_countermeasures("wiki","obfuscation",3000,test_times=10,legend=False)
    # generate_against_countermeasures("wiki","obfuscation",5000,test_times=10,legend=False)
    
    
    # generate_against_countermeasures("wiki","padding",3000,test_times=10,legend=False)
    # generate_against_countermeasures("wiki","padding",5000,test_times=10,legend=False)
    # generate_against_countermeasures("wiki","padding",1000,test_times=10,legend=True)
   

    # generate_against_countermeasures("enron","padding_cluster",1000,test_times=30,legend=True)
    # generate_against_countermeasures("lucene","padding_cluster",1000,test_times=30,legend=False)

    # generate_against_countermeasures("enron","padding_cluster",3000,test_times=10,legend=False)
    # generate_against_countermeasures("nytimes","padding_cluster",3000,test_times=10,legend=False)

    # generate_against_countermeasures("enron","padding_seal",3000,test_times=30,legend=False)
    # generate_against_countermeasures("nytimes","padding_seal",3000,test_times=30,legend=False)

    # generate_against_countermeasures("lucene","padding_seal",1000,test_times=30,legend=False)
    

    # generate_against_countermeasures("wiki","padding_seal",1000,test_times=10,legend=False)
    # generate_against_countermeasures("wiki","padding_cluster",1000,test_times=10,legend=False)

    # generate_against_countermeasures("wiki","padding_seal",3000,test_times=10,legend=False)
    # generate_against_countermeasures("wiki","padding_cluster",3000,test_times=10,legend=False)

    # generate_against_countermeasures("wiki","padding_seal",5000,test_times=10,legend=False)
    # generate_against_countermeasures("wiki","padding_cluster",5000,test_times=10,legend=False)

   