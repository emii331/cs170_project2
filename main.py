import numpy as np
from decimal import Decimal, getcontext

# test dataset paths
small_dataset_102 = "CS170_Small_Data__102.txt" # 3 1
large_dataset_45 = "CS170_Large_Data__45.txt" # 16 23
small_dataset_1 = "CS170_Small_Data__1.txt" # 5 3
small_dataset_2 = "CS170_Small_Data__2.txt" # 4 5
large_dataset_12 = "CS170_Large_Data__12.txt" # 26 38
large_dataset_17 = "CS170_Large_Data__17.txt" # 18 40

def main():
  print("Welcome to Emily's Feature Selection Algorithm!\n")
  dataset_selected = input("Type in the name of the file to test: ")
  dataset = np.loadtxt(dataset_selected)
  search_type_selected = input("\nType the number of the algorithm you want to run.\n   1) Forward Selection\n   2) Backward Elimination\n\n   Choice: ")
  
  num_samples = dataset.shape[0]
  num_features = dataset.shape[1] - 1
  print("\nThis dataset has " + str(num_features) + " features (not including the class attribute), with " + str(num_samples) + " instances.\n")
  accuracy_with_all_features = leave_one_out_validation(dataset, list(range(1, num_features+1)), None)
  print("Running nearest neighbor with all " + str(num_features) + " features, using \"leaving-one-out\" evaluation, I get an accuracy of " + str(accuracy_with_all_features*100) + "%\n")

  print("Beginning search.\n")
  if search_type_selected == '1':
    feature_search_forward(dataset)
  elif search_type_selected == '2':
    feature_search_backward(dataset)

  return

def feature_search_forward(data):
  best_feature_set = []
  class_column = data[:,0]
  class_1_count = np.count_nonzero(class_column == 1)
  class_2_count = np.count_nonzero(class_column == 2)
  default_rate = max(class_1_count, class_2_count) / len(data)
  best_overall_accuracy = default_rate
  print(default_rate)

  current_set_of_features = []
  for i in range(1,data.shape[1]):
    feature_to_add_at_this_level = []
    best_accuracy_at_curr_depth = 0
    for k in range(1,data.shape[1]):
      if not k in current_set_of_features:
        accuracy = leave_one_out_validation(data, current_set_of_features, k)

        set_to_print = current_set_of_features.copy()
        set_to_print.append(k)
        print("   Using feature(s) " + str(set_to_print) + " accuracy is " + str(accuracy*100) + "%")
        set_to_print.pop()
        
        if accuracy > best_accuracy_at_curr_depth:
          best_accuracy_at_curr_depth = accuracy
          feature_to_add_at_this_level = k

    current_set_of_features.append(feature_to_add_at_this_level)

    print("Feature set " + str(current_set_of_features) + " was best, accuracy is " + str(best_accuracy_at_curr_depth*100) + "%")

    if best_accuracy_at_curr_depth > best_overall_accuracy:
      best_overall_accuracy = best_accuracy_at_curr_depth
      best_feature_set = current_set_of_features.copy()
    elif best_accuracy_at_curr_depth < best_overall_accuracy and not len(current_set_of_features) == (data.shape[1] - 1):
      print("(Warning, accuracy has decreased! Continuing search in case of local maxima)")

    print("\n")


  print("Finished search!! The best feature subset is " + str(best_feature_set) + ", which has an accuracy of " + str(best_overall_accuracy*100) + "%")


def feature_search_backward(data):
  best_feature_set = list(range(1, data.shape[1]))
  best_overall_accuracy = leave_one_out_validation(data, best_feature_set, None)

  current_set_of_features = list(range(1, data.shape[1]))
  for i in range(1,data.shape[1]):
    feature_to_remove_at_this_level = []
    best_accuracy_at_curr_depth = 0
    for k in range(1,data.shape[1]):
      if k in current_set_of_features:
        current_set_of_features.remove(k)
        accuracy = leave_one_out_validation(data, current_set_of_features, None)

        set_to_print = current_set_of_features.copy()
        print("   Using feature(s) " + str(current_set_of_features) + " accuracy is " + str(accuracy*100) + "%")
        current_set_of_features.append(k)

        if accuracy > best_accuracy_at_curr_depth:
          best_accuracy_at_curr_depth = accuracy
          feature_to_remove_at_this_level = k
    
    print("Removing feature " + str(feature_to_remove_at_this_level) + " from feature set " + str(current_set_of_features) + " was best, accuracy is " + str(best_accuracy_at_curr_depth*100) + "%")
    current_set_of_features.remove(feature_to_remove_at_this_level)


    if best_accuracy_at_curr_depth > best_overall_accuracy:
      best_overall_accuracy = best_accuracy_at_curr_depth
      best_feature_set = current_set_of_features.copy()
    elif best_accuracy_at_curr_depth < best_overall_accuracy and not len(current_set_of_features) == 0:
      print("(Warning, accuracy has decreased! Continuing search in case of local maxima)")

    print("\n")


  print("Finished search!! The best feature subset is " + str(best_feature_set) + ", which has an accuracy of " + str(best_overall_accuracy*100) + "%")

def leave_one_out_validation(dataset, current_set, feature_to_add):
  data = dataset.copy()
  for feature in range(1,data.shape[1]):
    if not feature in current_set and not feature == feature_to_add:
      data[:,feature] = 0

  number_correctly_classified = 0

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
  
  getcontext().prec = 3
  accuracy = Decimal(number_correctly_classified / len(data))
  return accuracy


if __name__ == "__main__":
    main()