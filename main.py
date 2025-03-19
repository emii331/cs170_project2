import numpy as np
import os

# defining datapaths
small_dataset_102 = "CS170_Small_Data__102.txt"
large_dataset_45 = "CS170_Large_Data__45.txt"
large_dataset_12 = "CS170_Large_Data__12.txt"
large_dataset_17 = "CS170_Large_Data__17.txt"

def main():
  dataset = np.loadtxt(large_dataset_45)
  current_set_of_features = [16, 39]
  feature_to_add = 23

  accuracy = leave_one_out_validation(dataset, current_set_of_features, feature_to_add)
  print("Accuracy is " + str(accuracy))


  return

def leave_one_out_validation(data, current_set, feature_to_add):
  number_correctly_classified = 0
  print(data.shape[1])
  for feature in range(1,data.shape[1]):
    if not feature in current_set and not feature == feature_to_add:
      print("omitting feature " + str(feature))
      data[:,feature] = 0

  for i in range(0,len(data)):
    object_to_classify = data[i,1:]
    label_object_to_classify = data[i,0]

    nearest_neighbor_distance = np.inf
    nearest_neighbor_location = np.inf

    for k in range(0,len(data)):
      if k != i:
        distance = np.sqrt(np.sum((object_to_classify - data[k,1:]) ** 2))
        if distance < nearest_neighbor_distance:
          nearest_neighbor_distance = distance
          nearest_neighbor_location = k
          nearest_neighbor_label = data[nearest_neighbor_location,0]
    
    if label_object_to_classify == nearest_neighbor_label:
      number_correctly_classified = number_correctly_classified + 1
    
  accuracy = round(number_correctly_classified / len(data), 3)
  return accuracy




if __name__ == "__main__":
    main()