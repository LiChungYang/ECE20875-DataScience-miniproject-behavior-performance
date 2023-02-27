import pandas
import Behavior as bvr
import Cluster as cls
import GMM as gmm
import KMeans as kmean
import Helper as hlp
import Linear_Regression as lr
from sklearn.model_selection import train_test_split

'''
 The following is the starting code for path2 for data reading to make your first step easier.
 'dataset_2' is the clean data for path2.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['userID']      = df['userID'].astype(str)
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_2 = df
# print(dataset_2[0:35].to_string()) #This line will print out the first 35 rows of your data
# print(dataset_2[0:35]['numPauses'])#This line will print out the first 35 rows of number of Pauses field

# Problem 1: How well can the students be naturally grouped or clustered by their video-watching behavior 
# (fracSpent, fracComp, fracPaused, numPauses, avgPBR, numRWs, and numFFs)? 
# You should use all students that complete at least five of the videos in your analysis. 
# Hints: Would KMeans or Gaussian Mixture Models be more appropriate? Consider using both and comparing.

# get students fulfilling the criteria, video completion rate >= 0.9 with at least 5 videos watched
print("\nProblem 1\n",
      "============================================")
[vid_list, vid_data] = hlp.completedFiveVid(dataset_2)
behaviorMatrix = hlp.makeBehaviorMatrix(vid_data)

# group them by their behavior
behaviorList = bvr.makePointList(behaviorMatrix)

# kmean
# find best k with the shilouette score
k = kmean.best_k(behaviorMatrix)  
centroids = kmean.findCentroids(behaviorMatrix, k)
for i in range(len(centroids)):
    print("Centroid", i, ":", centroids[i])

# gmm 
# best no. of clusters
best_no_clustsers = gmm.gaus_mixture(behaviorMatrix)
print("Best number of clusters: ", best_no_clustsers)

# Problem 2: Can student's video-watching behavior be used to predict a student's performance (i.e., average score s across all quizzes)?
# (hint: Just choose 1 - 4 data fields to create your model. We are looking at your approach rather than model performance.)
print("\nProblem 2\n",
      "============================================")
# Using data fields: fracComp, numRWs
# use ridge regression to predict s
data = dataset_2.loc[:, ['fracComp','numRWs', 's']]
model_best = lr.regression(data)
for i in range(len(model_best.coef_)):
    print("feature_" , i, " * ", model_best.coef_[i])
print("Intercept: ", model_best.intercept_)

predictionNo = int(input("\nEnter the number of the students you want to predict: "))
for vidID in vid_list:
    print("\nVideo ", vidID, " Random Student Prediction(based on model): ")
    correctNo = hlp.predictRandomScore(dataset_2, vidID, None, None, model_best, "Model", predictionNo)
    percent = correctNo/predictionNo*100
    print("Out of ", predictionNo, " predictions, ", correctNo, " are correct -> ", percent, "%")

# Problem 3: Taking this a step further, how well can you predict a student's performance on a particular 
# in-video quiz question (i.e., whether they will be correct or incorrect) based on their video-watching behaviors 
# while watching the corresponding video? You should use all student-video pairs in your analysis.
print("\nProblem 3\n",
      "============================================")
vid_list = list(set(list(dataset_2['VidID'])))

# find nearest center average score s
centroids_0 = []
centroids_1 = []
testMatrices = {}
ways = []
for i in vid_list:
    centroid_0, centroid_1, testMatrix = hlp.getCentroidsForPrediction(dataset_2, i)
    testMatrices[i] = testMatrix
    centroids_0.append(centroid_0)
    centroids_1.append(centroid_1)
    clusterS, modelS = hlp.predictionScore(testMatrix, centroid_0, centroid_1, model_best)
    print("Accuracy of Prediction - Vid ", i, ": ", max(clusterS, modelS)*100, "% are corrected predicted")
    if(max(clusterS, modelS) == clusterS):
        ways.append("Cluster")
    else:
        ways.append("Model")

predictionNo = int(input("\nEnter the number of the students you want to predict: "))
for vidID in vid_list:
    print("\nVideo ", vidID, " Random Student Prediction(based on", ways[vidID], "):")
    correctNo = hlp.predictRandomScore(dataset_2, vidID, centroids_0, centroids_1, model_best, ways[vidID], predictionNo)
    percent = correctNo/predictionNo*100
    print("Out of ", predictionNo, " predictions, ", correctNo, " are correct -> ", percent, "%")
