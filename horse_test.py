import numpy as np
import math, string, copy

data = np.array([])
total_size = 0

# read file function
def read(file):
    global data
    global total_size
    f = open(file)
    for line in f:
        next_example = [line.rstrip() for line in line.split(",")]
        i = 0
        while i < 16:
            x = float(next_example[i])
            data = np.append(data, x)
            i += 1
        if next_example[-1] == 'colic.':
            data = np.append(data, False)
        elif next_example[-1] == 'healthy.':
            data = np.append(data, True)
        total_size += 1
    f.close()

# Colic goes over the array and count number of colic horses
def Colic(array):
    length = len(array)
    colic = 0
    i = 16
    while i < length:
        if array[i] == False:
            colic += 1
        i += 17
    return colic

# Healthy goes over the array and count number of healthy horses
def Healthy(array):
    length = len(array)
    healthy = 0
    i = 16
    while i < length:
        if array[i] == True:
            healthy += 1
        i += 17
    return healthy

# calculate entropy giving total horses with number of colic and healthy horses
def Entropy(colic, healthy, size):
    colic_rate = colic / size
    healthy_rate = healthy / size
    if colic == 0 and healthy == 0:
        return 0
    elif colic == 0:
        return -(healthy_rate * math.log(healthy_rate, 2))
    elif healthy == 0:
        return -(colic_rate * math.log(colic_rate, 2))
    else:
        return -(colic_rate * math.log(colic_rate, 2) + healthy_rate * math.log(healthy_rate, 2))

# sort and unique all the number of attribute[index] of all horses in array and return them in an array
def Threshold(array, index):
    new = np.array([])
    length = len(array)
    i = 0
    while i + index < length:
        number = array[i + index]
        new = np.append(new, number)
        i += 17
    new = np.sort(np.unique(new))
    return new

# split the array into two parts using threshold and index of threshold's attribute, calculate
# how many colic and healthy horses are under or above threshold and adds them up and return
# all those data in an array
def Split(array, threshold, index):
    less = 0
    more = 0
    less_colic = 0
    less_healthy = 0
    more_colic = 0
    more_healthy = 0
    length = len(array)
    i = 0
    while i + index < length:
        number = array[i + index]
        if number < threshold:
            less += 1
            if array[i + 16] == False:
                less_colic += 1
            else:
                less_healthy += 1
        else:
            more += 1
            if array[i + 16] == False:
                more_colic += 1
            else:
                more_healthy += 1
        i += 17
    new = np.array([less, less_colic, less_healthy, more, more_colic, more_healthy])
    return new

# calculate the remainder using all the given data
def Remainder(less, less_colic, less_healthy, more, more_colic, more_healthy):
    total = less + more
    lh_rate = less_healthy / less
    mh_rate = more_healthy / more
    
    less_than = 0
    more_than = 0
    
    if (lh_rate != 1 and lh_rate != 0):
        less_than = (less / total) * (-(lh_rate * math.log(lh_rate, 2) + ((1 - lh_rate) * math.log(1 - lh_rate, 2))))
    if (mh_rate != 1 and mh_rate != 0):
        more_than = (more / total) * (-(mh_rate * math.log(mh_rate, 2) + ((1 - mh_rate) * math.log(1 - mh_rate, 2))))
    return less_than + more_than

# calculate the best information gain given an array of data and index of the attribute we want to find
def IG(array, index):
    threshold_array = Threshold(array, index)
    length = len(threshold_array)
    size = len(array)
    i = 0
    best_threshold = -1
    best_IG = -1
    best_index = -1
    # loop through all thresholds
    while i < length - 1:
        threshold = (threshold_array[i] + threshold_array[i+1]) / 2
        new_array = Split(array, threshold, index)
        less = new_array[0]
        less_colic = new_array[1]
        less_healthy = new_array[2]
        more = new_array[3]
        more_colic = new_array[4]
        more_healthy = new_array[5]
        
        colic_number = Colic(array)
        healthy_number = Healthy(array)
        total = more + less
        
        #calculate all entropy and remainder of each threshold
        entropy = Entropy(colic_number, healthy_number, total)
        remainder = Remainder(less, less_colic, less_healthy, more, more_colic, more_healthy)
        IG = entropy - remainder
        
        #record the best information gain, best threshold, and its index among all thresholds and return it
        if best_IG == -1 and best_threshold == -1 and best_index == -1:
            best_IG = copy.copy(IG)
            best_threshold = copy.copy(threshold)
            best_index = copy.copy(index)
        elif best_IG < IG:
            best_IG = copy.copy(IG)
            best_threshold = copy.copy(threshold)
            best_index = copy.copy(index)
        i += 1
    array = np.array([best_IG, best_threshold, best_index])
    return array

# return whether the array is empty
def is_empty(array):
    if len(array) == 0:
        return True
    else:
        return False

# return whether the horses in the array are all colic or all healthy
def all_same(array):
    attribute = ''
    i = 16
    length = len(array)
    while i < length:
        if attribute == '':
            attribute = copy.copy(array[i])
        elif attribute != copy.copy(array[i]):
            return False
        i += 17
    return attribute


class Node:
    def __init__(self, variable, left_node, right_node):
        self.variable = variable;
        self.left_node = left_node;
        self.right_node = right_node;


# the decision tree function
def DTL(example, attribute, default):
    # base case 1 that array is empty
    if is_empty(example):
        return default
    
    # base case 2 that all horses in the array are colic or healthy
    attribute = -1
    i = 16
    length = len(example)
    j = 0
    while i < length:
        if attribute == -1:
            attribute = example[i]
        if attribute != example[i]:
            j = 1
            break
        i += 17
    if j == 0:
        return attribute
    
    # case 3 where we need to find the best IG and seperate data into left and right
    # subtree and recursion on them
    best_IG = 0
    best_threshold = 0
    best_index = 0
    for i in range(0, 16):
        array = IG(example, i)
        current_IG = array[0]
        current_threshold = array[1]
        current_index = int(array[2])
        if current_IG > best_IG:
            best_IG = current_IG
            best_threshold = current_threshold
            best_index = current_index
    lower_examples = np.array([])
    upper_examples = np.array([])
    i = 0
    length = len(example)
    while i + best_index < length:
        if example[i + best_index] > best_threshold:
            for j in range(0, 17):
                upper_examples = np.append(upper_examples, example[i + j])
        else:
            for j in range(0, 17):
                lower_examples = np.append(lower_examples, example[i + j])
        i += 17
    subtree_left = DTL(lower_examples,1,1)
    subtree_right = DTL(upper_examples,1,1)

    tree = Node(best_index,subtree_left, subtree_right)
    return tree

def main():
    global data
    global total_size
    
    read("horseTrain.txt")
    tree = DTL(data, 1, 1)
    return tree
    

main()