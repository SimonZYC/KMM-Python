import os
import sys
import numpy as np

execNames=['k_means','kernel_k_means','spectral','KMM']
datasets=['birch','dim','overlap','shape','unbalance']

setFilePath = "./data"
desFilePath = "./EXP"

savepath=os.path.join(desFilePath,'result')
result=open(savepath,'w')

for dsets in datasets:
    # group experiments
    result.write('%s---------------------------------\n'%dsets)
    setspath=os.path.join(setFilePath,dsets)
    sets=set()
    for file in os.listdir(setspath):   # get unique datasets
        sets.add(file.split('.')[0])
    # get unique file name
    for file in sets:
        result.write('  %s: Purity  NMI Time\n'%file)

        for execName in execNames:
            p=os.path.join(desFilePath,execName,file)

            purity=[]
            nmi=[]
            time=[]
            if not os.path.exists(p):
                result.write('%s:  -  -  -\n'%(execName))
                continue
            with open(p,'r') as f:
                while True:
                    line=f.readline().strip()
                    if not line:
                        break
                    #print(line)
                    line=line.split('=')
                    
                    handle=line[0]
                    value=line[1]
                    if handle=='Purity':
                        purity.append(float(value)*100)
                    elif handle=='NMI':
                        nmi.append(float(value)*100)
                    elif handle=='time':
                        time.append(float(value))

            f.close()

            if len(purity)!= 0:

                p_mean=np.mean(purity)
                p_std=np.std(purity)

                n_mean=np.mean(nmi)
                n_std=np.std(nmi)

                t_mean=np.mean(time)
                t_std=np.std(time)

                result.write('%s:  %f  %f  %f\n'%(execName,p_mean,n_mean,t_mean))
            else:
                result.write('%s:  -  -  -\n'%(execName))


        result.write('\n')
result.close()


