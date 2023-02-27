import numpy as np
import random

from scipy import rand
import KMeans as kmean
from sklearn.model_selection import train_test_split

# Function: get_by_VidID()
# Input   : 
#           behavior-performance.txt as a Pandas Dataframe
# Output  : 
#           vid_list (list) : A list of all VidIDs in the dataframe
#           vid_data (dict) : A dictionary, where the key is a VidID and Value is a Dataframe with ONLY the values for the selected VidID

def get_by_VidID(df):

  vid_list = list(set(list(df['VidID'])))
  vid_data = dict()
  for element in vid_list:
    vid_data[element] = df[df['VidID']==element]
  return vid_list,vid_data

def completedFiveVid(df):

  vid_list = list(set(list(df['VidID'])))
  vid_data = df[df['fracComp']>=0.9]
  vid_data['count'] = vid_data.groupby(['userID'])['userID'].transform('count')
  vid_data = vid_data[vid_data['count']>=5]
  vid_data = vid_data.drop(['count'],axis=1)
  return vid_list,vid_data

def makeBehaviorMatrix(df):

  behaviorMatrix = np.array(df.loc[:,['fracSpent','fracComp','fracPaused','numPauses','avgPBR','numRWs','numFFs', 's']])
  return behaviorMatrix

def getCentroidsForPrediction(dataset_2, vidid):
  dataset = dataset_2[dataset_2['VidID']==vidid]
  data =  np.array(dataset.loc[:, ['fracComp','numRWs', 's']])
  testMatrix, trainMatrix = train_test_split(data, test_size=0.2)
  trainMatrix_0 = []
  trainMatrix_1 = []
  for row in range(len(data)):
    if(data[row][2] == 0):
      trainMatrix_0.append(data[row])
    else:
      trainMatrix_1.append(data[row])
  
  k_1 = kmean.best_k(trainMatrix_1)
  k_0 = kmean.best_k(trainMatrix_0)
   
  centroids_1 = kmean.findCentroids(trainMatrix_1, k_1)
  centroids_0 = kmean.findCentroids(trainMatrix_0, k_0)
  
  return centroids_0, centroids_1, testMatrix

# calculate the distance from the data to all the centroids
# perform fractional distance calculation
# distance to centroid of zeros / distance to centroid of ones + distance to centroid of zeros
# add all the distances and return the list
def distanceScore(centroid_0, centroid_1, data):
  scores = []
  for row in data:
    c0toc1_matrix = []
    dist_0_list = []
    dist_1_list = []
    for i in range(len(centroid_0)):
      temp = []
      for j in range(len(centroid_1)):
        temp.append(np.linalg.norm(centroid_0[i]-centroid_1[j]))
      c0toc1_matrix.append(temp)
    for c0 in centroid_0:
      dist_0_list.append(np.linalg.norm(c0-row))
    for c1 in centroid_1:
      dist_1_list.append(np.linalg.norm(c1-row))
    score = 0.0
    total = 0 
    for i in range(len(c0toc1_matrix)):
      for j in range(len(c0toc1_matrix[i])):
        score += (dist_0_list[i] / (dist_0_list[i] + dist_1_list[j]))
        total += 1
    scores.append(score/total)
  return scores

# predict the score for a student based on the kmeans model and ridge regression model
# use the fractional distance from each data to all the centroids to perform prediction
# and then use the distances from each centroid to the data to predict the score
# find out all the predictions for the student and then find the average correctness or accuracy of the predictions for both kmean and ridge method
# return the accuracy of the prediction
def predictionScore(matrix, centroid_0, centroid_1, model):
  features = np.array(matrix[:,0:1])
  targets = np.array(matrix[:,2])

  scores = distanceScore(centroid_0, centroid_1, features)

  s_list_1 = []
  for s in scores: 
    if(s >= 0.5):
      s_list_1.append(1)
    else: 
      s_list_1.append(0)
  predictionAccuracy = 0
  for i in range(len(s_list_1)):
    if(targets[i] == s_list_1[i]):
      predictionAccuracy += 1
  
  s_list_2 = []
  tempList =[]
  for row in features:
    tempS = 0.0
    for i in range(len(row)):
      tempS += row[i]*model.coef_[i]
    tempList.append(tempS+model.intercept_)
  for s in tempList:
    if(s >= 0.5):
      s_list_2.append(1)
    else:
      s_list_2.append(0)
  modelAccuracy = 0
  for i in range(len(s_list_2)):
    if(targets[i] == s_list_2[i]):
      modelAccuracy += 1
  
  return predictionAccuracy/len(scores), modelAccuracy/len(scores)

# randomly select n students from the dataset and perform prediction
def predictRandomScore(df, vidID, centroid_0, centroid_1, model, way, predictionNo):
  randomStudent = []
  temp = df[df['VidID']==vidID]
  
  data = np.array(temp.loc[:,['userID', 'fracComp', 'numRWs', 's']])
  features = np.array(temp.loc[:,['fracComp', 'numRWs']])
  
  for i in range(predictionNo):
    idx = random.randint(0, len(features[:,0])-1)
    if(way == 'Cluster'):
      # this is feature is 1D so the distacneScore will return the number of columns, since we don't have rows for 1D
      score1, score2 = distanceScore(centroid_0, centroid_1, features[idx, :])
      score = score1 + score2 / 2
    else:
      score = sum([features[idx][k]*model.coef_[k] for k in range(len(features[idx,:]))]) + model.intercept_
    if(score >= 0.5):
      randomStudent.append((data[:,0][idx], 1, data[:,3][idx]))
    else:
      randomStudent.append((data[:,0][idx], 0, data[:,3][idx]))
    
  # count the number of correct predictions
  correct = 0
  for element in randomStudent:
    if(element[1] == element[2]):
      correct += 1

  return correct
  