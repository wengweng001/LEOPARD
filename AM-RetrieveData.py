import pickle
import numpy as np

Source = 'AM1'
Target = 'AM4'
TypeRun='Init'

whichRound=1
indAccDisplay=0
idxPerformance=0
indexPerformanceHistory=1
total=0
nRound=5
accuracy=[]

# 'Results/'+Source+'/'+Target+'/'+TypeRun+'Result{}'.format(iRoundAblation+1)

for iRoundTest in range(nRound):
    # string = Source + Target +  'Ablation{}'.format(iRoundTest)
    string='Results/'+Source+'/'+Target+'/'+Source+Target+TypeRun+'Result{}'.format(iRoundTest)
    file = open(string, 'rb')
    # object_file = pickle.load(file)
    object_file = pickle.load(file)
    print('Index ', iRoundTest, ' ', object_file[0][0][0])
    # total+=object_file[0][0][0]
    accuracy.append(object_file[0][0][0])
    if iRoundTest==nRound-1:
        print("Average Accuracy"+Source+Target+TypeRun,np.mean(accuracy),"Standard deviation : ",np.std(accuracy))
    file.close()

# accuracy=object_file[idxPerformance][whichRound][indAccDisplay]


# object_file[0][0][0] ==> accuracy


print('Preq Accuracy: ', meanResults[0].item(), '(+/-)', stdResults[0].item())
print('ARI: ', meanResults[1].item(), '(+/-)', stdResults[1].item())
print('NMI: ', meanResults[2].item(), '(+/-)', stdResults[2].item())
print('F1 score: ', meanResults[3].item(), '(+/-)', stdResults[3].item())
print('Precision: ', meanResults[4].item(), '(+/-)', stdResults[4].item())
print('Recall: ', meanResults[5].item(), '(+/-)', stdResults[5].item())
print('Training time: ', meanResults[6].item(), '(+/-)', stdResults[6].item())
print('Testing time: ', meanResults[7].item(), '(+/-)', stdResults[7].item())
print('\n')

print('========== Network ==========')
print('Number of hidden layers: ', meanResults[8].item(), '(+/-)', stdResults[8].item())
print('Number of features: ', meanResults[9].item(), '(+/-)', stdResults[9].item())
print('Number of clusters: ', meanResults[10].item(), '(+/-)', stdResults[10].item())
