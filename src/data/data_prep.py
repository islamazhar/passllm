import os 




file_name_in = os.getenv("HOME") + "/pwdata/data/new_mixed_full/mixed_full_leak_data_80.txt"
fin = open(file_name_in)


file_name_out = os.getenv("HOME") + "/pwdata/mazharul/llama2-model-train-pws/train.txt"


len_check = lambda pw: len(pw) >=4 and len(pw) <=30 



# filter usernames, and passwords
with open(file_name_out, 'w') as fout:
    lineNo = 0
    for count, line in enumerate(fin):
        # print(line)
        
        try:
            uname, pws = line.strip().split('\t', 1)
            # print(pws)
            upws = {x for x in pws.split('\t') if len_check(x)}
            # print(uname, upws)
            if len(upws) > 1:
                fout.write('\t'.join(upws) + '\n')
                lineNo += 1
                if lineNo % 100000000 == 0:
                    print(upws)
        except :
            continue 
            # if lineNo > 30: 
            #     break
            
            
# prepare the file 
import random 

file_name_out2 = os.getenv("HOME") + "/pwdata/mazharul/llama2-model-train-pws/pre_train.txt"
with open(file_name_out, 'r') as fin, open(file_name_out2, 'w') as fout:
    for line in fin:
        pws = line.strip().split('\t')
        pw_selected = random.choice(pws)
        pws.remove(pw_selected)
        pws =  '\t'.join(pws)
        fout.write(f"{pw_selected}\t{pws}")
        fout.write("\n")        