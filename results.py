
import numpy as np
import pandas as pd
import pickle
import os


params = {
    'no_of_epoch_clustering': [1, ],
    'no_of_epoch_domain_adaptation': [1, ],
    # 'learning rate': [0.001, 0.01, 0.1, 1],
    'alpha_kl_divergence':     [0.0, 0.001, 0.001, 0.001, 0.001, 0.01,  0.01, 0.01, 0.01, 0.1  , 0.1,  0.1, 0.1, 1.0  , 1.0,  1.0, 1.0],
    'alpha_domain_adaptation': [0.0, 0.001, 0.01 , 0.1,   1.0,   0.001, 0.01, 0.1,  1.0,  0.001, 0.01, 0.1, 1.0, 0.001, 0.01, 0.1, 1.0],
}

file_path = 'performance'

for alpha1,alpha2 in zip(params['alpha_kl_divergence'], params['alpha_domain_adaptation']):
    for noOfEpochPrequential in params['no_of_epoch_clustering']:
        for nEpochKLL1 in params['no_of_epoch_domain_adaptation']:
            #========Setting========
            Source = ['AM1', 'AM1', 'AM1', 'AM1', 'AM2', 'AM2', 'AM2', 'AM2', 'AM3', 'AM3', 'AM3', 'AM3', 'AM4', 'AM4', 'AM4', 'AM4', 'AM5', 'AM5', 'AM5', 'AM5',]
            Target = ['AM2', 'AM3', 'AM4', 'AM5', 'AM1', 'AM3', 'AM4', 'AM5', 'AM1', 'AM2', 'AM4', 'AM5', 'AM1', 'AM2', 'AM3', 'AM5', 'AM1', 'AM2', 'AM3', 'AM4',]

            Source = ['AM1', 'AM1', 'AM3', 'AM3', 'AM4', 'AM5', 'AM5', 'AM5', ]
            Target = ['AM3', 'AM4', 'AM1', 'AM5', 'AM1', 'AM1', 'AM2', 'AM3', ]
            df = pd.DataFrame(columns = ["TypeRun","Epoch-c","Epoch-d","alpha-kl", "alpha-dc", "accT","accS"])
            
            for s, t in zip(Source,Target):

                Init = 'Run'
                TypeRun= s+t+Init
                default=True

                if default: # This setting parameters are used in all experiments
                    nRoundTest = 5  # default 5 or 1 if you one to run 1 time
                    portion = 0.1;  # portion of labeled data in the warm up phase #DEFAULT 0.1, 0.2 UP LEAD TO ERROR DUE TO LACK OF LABELED DATA
                    nInitCluster = 2  # setting number of initial cluster # DEFAULT 2
                    noOfEpochInitializationCluster = 100  # number of epoch during cluster initialization #DEFAULT 1000
                    # noOfEpochPrequential = 5  # number of epoch during prequential (CNN and Auto Encoder) #DEFAULT 5
                    # nEpochKLL1 = 1  # number of epoch during prequential (KL and L1) # DEFAULT 1
                    # Drift Location=============
                    listDriftSource = [5]
                    listDriftTarget = [6]
                else: # This setting one to test
                    nRoundTest = 5  # default 5 or 1 if you one to run 1 time
                    portion = 0.1;  # portion of labeled data in the warm up phase #DEFAULT 0.1, 0.2 UP LEAD TO ERROR DUE TO LACK OF LABELED DATA
                    nInitCluster = 2  # setting number of initial cluster # DEFAULT 2
                    noOfEpochInitializationCluster = 100  # number of epoch during cluster initialization #DEFAULT 1000
                    # noOfEpochPrequential = 5  # number of epoch during prequential (CNN and Auto Encoder) #DEFAULT 5
                    # nEpochKLL1 = 1  # number of epoch during prequential (KL and L1) # DEFAULT 1
                    # Drift Location=============
                    listDriftSource = [5]
                    listDriftTarget = [6]

                paramstamp = ('minibatch' + TypeRun+str(nRoundTest)+'Times'+str(portion)+'Portion'
                +str(noOfEpochPrequential)+'ClusEpochs'+str(nEpochKLL1)+'DCEpochs'+str(alpha1)+'alphakl'+str(alpha2)+'alphadc')
                # paramstamp = (TypeRun + str(nRoundTest) + 'Times' + str(portion) + 'Portion'
                #               + str(noOfEpochPrequential) + 'ClusEpochs' + str(nEpochKLL1) + 'DCEpochs' + str(
                #             alpha1) + 'alphakl' + str(alpha2) + 'alphadc')

                resultfile = paramstamp+'.pkl'
                result_dir = file_path + '/' + resultfile
                if os.path.isfile(result_dir):
                    try:
                        w1, w2, w3, w4, w5, w6 = pickle.load(open(result_dir, 'rb'), encoding='utf-8')
                
                        # print('----------------------------------------------')
                        # print(paramstamp)
                        # print('epoch_clus\t', noOfEpochPrequential)
                        # print('epoch_dc\t', nEpochKLL1)
                        # print('alpha_clus\t', alpha1)
                        # print('alpha_dc\t', alpha2)
                        # print('Acc on T: %.4f +/- %.4f' % (np.mean(w1), np.std(w1)))
                        # print('F1 on T: %.4f +/- %.4f' % (np.mean(w5), np.std(w5)))
                        # print('Acc on S: %.4f +/- %.4f' % (np.mean(w2), np.std(w2)))
                        # print('Clusters: %.4f +/- %.4f' % (np.mean(41), np.std(w4)))

                        df = df.append({"TypeRun":TypeRun,"Epoch-c":noOfEpochPrequential,"Epoch-d":nEpochKLL1,
                        "alpha-kl":alpha1,"alpha-dc":alpha2,
                        "accT":'%.4f +/- %.4f' % (np.mean(w1), np.std(w1)),
                        "accS":'%.4f +/- %.4f' % (np.mean(w2), np.std(w2))}, ignore_index=True)

                        continue
                    except:
                        pass
                else:
                    pass

            if not df.empty:
                print(df[["TypeRun","Epoch-c","Epoch-d","alpha-kl", "alpha-dc", "accT"]])

print(df)