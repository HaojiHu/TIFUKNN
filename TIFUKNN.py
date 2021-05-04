from __future__ import unicode_literals, print_function, division
import numpy as np
import sys
import math
import csv


activate_codes_num = -1
next_k_step = 1
training_chunk = 0
test_chunk = 1




def add_history(data_history,training_key_set,output_size):
    sum_history = {}
    for key in training_key_set:
        sum_vector = np.zeros(output_size)
        count = 0
        for lst in data_history[key]:
            vec = np.zeros(output_size)
            for ele in lst:
                vec[ele] = 1
            if vec[-2] == 1 or vec[-1] == 1:
                continue
            sum_vector += vec
            count += 1
        sum_vector = sum_vector / count
        sum_history[key] = sum_vector
    return sum_history



def temporal_decay_add_history(data_set, key_set, output_size,within_decay_rate):
    sum_history = {}
    for key in key_set:
        vec_list = data_set[key]
        num_vec = len(vec_list) - 2
        his_list = np.zeros(output_size)
        for idx in range(1,num_vec+1):
            his_vec = np.zeros(output_size)
            decayed_val = np.power(within_decay_rate,num_vec-idx)
            for ele in vec_list[idx]:
                his_vec[ele] = decayed_val
            his_list += his_vec
        sum_history[key] = his_list/num_vec
        # sum_history[key] = np.multiply(his_list / num_vec, IDF)

    return sum_history


def KNN(query_set, target_set, k):
    history_mat = []
    for key in target_set.keys():
        history_mat.append(target_set[key])
    test_mat = []
    for key in query_set.keys():
        test_mat.append(query_set[key])
    # print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(test_mat)
    # print('Finish KNN search.' )
    return indices,distances


def weighted_aggragate_outputs(data_chunk,training_key_set,index,distance,output_size):
    output_vectors = []
    key_set = training_key_set
    for index_list_id in range(len(index)):
        outputs = []
        for vec_idx in range(1,next_k_step+1):

            target_vec_list = []
            weight_list = []
            for id in range(len(index[index_list_id])):
                dis = distance[index_list_id][id]
                if dis == 0:
                    weight_list.append(0)
                else:
                    weight_list.append(1 / dis)
            new_weight = softmax(weight_list)
            for i in range(len(new_weight)):
                if new_weight[i] == 0:
                    new_weight[i] = 1
            vec = np.zeros(output_size)
            for id in range(len(index[index_list_id])):
                idx = index[index_list_id][id]
                target_list = data_chunk[test_chunk][key_set[idx]][vec_idx]
                for ele in target_list:
                    vec[ele] += new_weight[id]
            outputs.append(vec)
        output_vectors.append(outputs)
    return output_vectors


from sklearn.neighbors import NearestNeighbors

def KNN_history_record1(sum_history, output_size,k):
    history_mat = []
    for key in sum_history.keys():
        history_mat.append(sum_history[key])

    print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(history_mat)
    KNN_history = {}
    key_set = list(sum_history)
    for id in range(len(key_set)):
#    for idx_list in indices:
        idx_list = indices[id]
        NN_history = np.zeros(output_size)
        for idx in idx_list:
            NN_history += sum_history[key_set[idx]]
        NN_history = NN_history / k
        KNN_history[key_set[id]] = NN_history



    return KNN_history

def KNN_history_record2(query_set, sum_history, output_size,k):
    history_mat = []
    for key in sum_history.keys():
        history_mat.append(sum_history[key])
    test_mat = []
    for key in query_set.keys():
        test_mat.append(query_set[key])
    print('Finding k nearest neighbors...')
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute').fit(history_mat)
    distances, indices = nbrs.kneighbors(test_mat)
    KNN_history = {}
    key_set = list(query_set)
    training_key_set = list(sum_history)
    for id in range(len(key_set)):
#    for idx_list in indices:
        idx_list = indices[id]
        NN_history = np.zeros(output_size)
        for idx in idx_list:
            NN_history += sum_history[training_key_set[idx]]
        NN_history = NN_history / k
        KNN_history[key_set[id]] = NN_history


    return KNN_history,indices
def group_history_list(his_list,group_size):
    grouped_vec_list = []
    if len(his_list) < group_size:
        #sum = np.zeros(len(his_list[0]))
        for j in range(len(his_list)):
            grouped_vec_list.append(his_list[j])

        return grouped_vec_list, len(his_list)
    else:
        est_num_vec_each_block = len(his_list)/group_size
        base_num_vec_each_block = int(np.floor(len(his_list)/group_size))
        residual = est_num_vec_each_block - base_num_vec_each_block

        num_vec_has_extra_vec = int(np.round(residual * group_size))

        if residual == 0:
            for i in range(group_size):
                if len(his_list)<1:
                    print('len(his_list)<1')
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i*base_num_vec_each_block+j >= len(his_list):
                        print('i*num_vec_each_block+j')
                    sum += his_list[i*base_num_vec_each_block+j]
                grouped_vec_list.append(sum/base_num_vec_each_block)
        else:

            for i in range(group_size - num_vec_has_extra_vec):
                sum = np.zeros(len(his_list[0]))
                for j in range(base_num_vec_each_block):
                    if i*base_num_vec_each_block+j >= len(his_list):
                        print('i*base_num_vec_each_block+j')
                    sum += his_list[i*base_num_vec_each_block+j]
                    last_idx = i * base_num_vec_each_block + j
                grouped_vec_list.append(sum/base_num_vec_each_block)

            est_num = int(np.ceil(est_num_vec_each_block))
            start_group_idx = group_size - num_vec_has_extra_vec
            if len(his_list) - start_group_idx*base_num_vec_each_block >= est_num_vec_each_block:
                for i in range(start_group_idx,group_size):
                    sum = np.zeros(len(his_list[0]))
                    for j in range(est_num):
                        # if residual+(i-1)*est_num_vec_each_block+j >= len(his_list):
                        #     print('residual+(i-1)*num_vec_each_block+j')
                        #     print('len(his_list)')
                        iidxx = last_idx + 1+(i-start_group_idx)*est_num+j
                        if  iidxx >= len(his_list) or iidxx<0:
                            print('last_idx + 1+(i-start_group_idx)*est_num+j')
                        sum += his_list[iidxx]
                    grouped_vec_list.append(sum/est_num)

        return grouped_vec_list, group_size

def temporal_decay_sum_history(data_set, key_set, output_size,group_size,within_decay_rate,group_decay_rate):
    sum_history = {}
    for key in key_set:
        vec_list = data_set[key]
        num_vec = len(vec_list) - 2
        his_list = []
        for idx in range(1,num_vec+1):
            his_vec = np.zeros(output_size)
            decayed_val = np.power(within_decay_rate,num_vec-idx)
            for ele in vec_list[idx]:
                his_vec[ele] = decayed_val
            his_list.append(his_vec)

        grouped_list,real_group_size = group_history_list(his_list,group_size)
        his_vec = np.zeros(output_size)
        for idx in range(real_group_size):
            decayed_val = np.power(group_decay_rate, group_size - 1 - idx)
            if idx>=len(grouped_list):
                print( 'idx: '+ str(idx))
                print('len(grouped_list): ' + str(len(grouped_list)))
            his_vec += grouped_list[idx]*decayed_val
        sum_history[key] = his_vec/real_group_size
        # sum_history[key] = np.multiply(his_vec / real_group_size, IDF)
    return sum_history

def partition_the_data(data_chunk,key_set):
    filtered_key_set = []
    for key in key_set:
        if len(data_chunk[training_chunk][key])<=3:
            continue
        if len(data_chunk[test_chunk][key])<2+next_k_step:
            continue
        filtered_key_set.append(key)

    training_key_set = filtered_key_set[0:int(4 / 5 * len(filtered_key_set))]
    print(len(training_key_set))
    test_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)):]
    return training_key_set,test_key_set


def partition_the_data_validate(data_chunk, key_set, next_k_step):
    filtered_key_set = []
    past_chunk = 0
    future_chunk = 1
    for key in key_set:
        if len(data_chunk[past_chunk][key]) <= 3:
            continue
        if len(data_chunk[future_chunk][key]) < 2 + next_k_step:
            continue
        filtered_key_set.append(key)

    training_key_set = filtered_key_set[0:int(4 / 5 * len(filtered_key_set)*0.9)]
    validation_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)*0.9):int(4 / 5 * len(filtered_key_set))]
    print('Number of training instances: ' + str(len(training_key_set)))
    test_key_set = filtered_key_set[int(4 / 5 * len(filtered_key_set)):]
    return training_key_set, validation_key_set, test_key_set

def most_frequent_elements(data_chunk,index,training_key_set,output_size):
    output_vectors = []

    for vec_idx in range(1,next_k_step+1):
        vec = np.zeros(output_size)
        for idx in index:
            target_vec = data_chunk[test_chunk][training_key_set[idx]][vec_idx]
            for ele in target_vec:
                vec[ele] += 1

        output_vectors.append(vec)
    return output_vectors

def predict_with_elements_in_input(sum_history,key):
    output_vectors = []

    for idx in range(next_k_step):
        vec = sum_history[key]
        output_vectors.append(vec)
    return output_vectors


def generate_dictionary_BA(files, attributes_list):
    # path = '../Minnemudac/'
    #files = ['Coborn_history_order.csv','Coborn_future_order.csv']
    #files = ['BA_history_order.csv', 'BA_future_order.csv']
    #attributes_list = ['MATERIAL_NUMBER']
    dictionary_table = {}
    counter_table = {}
    for attr in attributes_list:
        dictionary = {}
        dictionary_table[attr] = dictionary
        counter_table[attr] = 0

    csv.field_size_limit(sys.maxsize)
    for filename in files:
        count = 0
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if count == 0:
                    count += 1
                    continue
                key = attributes_list[0]
                if row[2] not in dictionary_table[key]:
                    dictionary_table[key][row[2]] = counter_table[key]
                    counter_table[key] = counter_table[key] + 1
                    count += 1

    print(counter_table)

    total = 0
    for key in counter_table.keys():
        total = total + counter_table[key]

    print('# dimensions of final vector: ' + str(total) + ' | '+str(count-1))

    return dictionary_table, total, counter_table

def read_claim2vector_embedding_file_no_vector(files):
    #attributes_list = ['DRG', 'PROVCAT ', 'RVNU_CD', 'DIAG', 'PROC']
    attributes_list = ['MATERIAL_NUMBER']
    # path = '../Minnemudac/'
    print('start dictionary generation...')
    dictionary_table, num_dim, counter_table = generate_dictionary_BA(files, attributes_list)
    print('finish dictionary generation*****')
    usr_attr = 'CUSTOMER_ID'
    ord_attr = 'ORDER_NUMBER'

    #dictionary_table, num_dim, counter_table = GDF.generate_dictionary(attributes_list)

    freq_max = 200
    ## all the follow three ways array. First index is patient, second index is the time step, third is the feature vector
    data_chunk = []
    day_gap_counter = []
    claims_counter = 0
    num_claim = 0
    code_freq_at_first_claim = np.zeros(num_dim+2)


    for file_id in range(len(files)):

        count = 0
        data_chunk.append({})
        filename = files[file_id]
        with open(filename, 'r') as csvfile:
            #gap_within_one_year = np.zeros(365)
            reader = csv.DictReader(csvfile)
            last_pid_date = '*'
            last_pid = '-1'
            last_days = -1
            # 2 more elements in the end for start and end states
            feature_vector = []
            for row in reader:
                cur_pid_date = row[usr_attr] + '_' + row[ord_attr]
                cur_pid = row[usr_attr]
                #cur_days = int(row[ord_attr])

                if cur_pid != last_pid:
                    # start state
                    tmp = [-1]
                    data_chunk[file_id][cur_pid] = []
                    data_chunk[file_id][cur_pid].append(tmp)
                    num_claim = 0
                # else:
                #     if last_days != cur_days and last_days != -1 and file_id != 0 and file_id != 2:
                #
                #         gap_within_one_year[cur_days - last_days] = gap_within_one_year[cur_days - last_days] + 1


                if cur_pid_date not in last_pid_date:
                    if last_pid_date not in '*' and last_pid not in '-1':
                        sorted_feature_vector = np.sort(feature_vector)
                        data_chunk[file_id][last_pid].append(sorted_feature_vector)
                        if len(sorted_feature_vector) > 0:
                            count = count + 1
                        #data_chunk[file_id][last_pid].append(feature_vector)
                    feature_vector = []

                    claims_counter = 0
                if cur_pid != last_pid:
                    # end state
                    if last_pid not in '-1':

                        tmp = [-1]
                        data_chunk[file_id][last_pid].append(tmp)

                key = attributes_list[0]

                within_idx = dictionary_table[key][row[key]]
                previous_idx = 0

                for j in range(attributes_list.index(key)):
                    previous_idx = previous_idx + counter_table[attributes_list[j]]
                idx = within_idx + previous_idx

                # set corresponding dimention to 1
                if idx not in feature_vector:
                    feature_vector.append(idx)

                last_pid_date = cur_pid_date
                last_pid = cur_pid
                #last_days = cur_days
                if file_id == 1:
                    claims_counter = claims_counter + 1


            if last_pid_date not in '*' and last_pid not in '-1':
                data_chunk[file_id][last_pid].append(np.sort(feature_vector))
        # if file_id != 0 and file_id != 2:
        #     day_gap_counter.append(gap_within_one_year)
        # print('num of vectors having entries more than 1: ' + str(count))
  #  print(len(data_chunk[0]))

    #print(data_chunk[0]['33050811449.0'])

#    print(data_chunk[0]['33051194484.0'])

 #   print(data_chunk[0]['33051313687.0'])

    return data_chunk, num_dim + 2, code_freq_at_first_claim


def get_precision_recall_Fscore(groundtruth,pred):
    a = groundtruth
    b = pred
    correct = 0
    truth = 0
    positive = 0

    for idx in range(len(a)):
        if a[idx] == 1:
            truth += 1
            if b[idx] == 1:
                correct += 1
        if b[idx] == 1:
            positive += 1

    flag = 0
    if 0 == positive:
        precision = 0
        flag = 1
        #print('postivie is 0')
    else:
        precision = correct/positive
    if 0 == truth:
        recall = 0
        flag = 1
        #print('recall is 0')
    else:
        recall = correct/truth

    if flag == 0 and precision + recall > 0:
        F = 2*precision*recall/(precision+recall)
    else:
        F = 0
    return precision, recall, F, correct

def get_F_score(prediction, test_Y):
    jaccard_similarity = []
    prec = []
    rec = []

    count = 0
    for idx in range(len(test_Y)):
        pred = prediction[idx]
        T = 0
        P = 0
        correct = 0
        for id in range(len(pred)):
            if test_Y[idx][id] == 1:
                T = T + 1
                if pred[id] == 1:
                    correct = correct + 1
            if pred[id] == 1:
                P = P + 1

        if P == 0 or T == 0:
            continue
        precision = correct / P
        recall = correct / T
        prec.append(precision)
        rec.append(recall)
        if correct == 0:
            jaccard_similarity.append(0)
        else:
            jaccard_similarity.append(2 * precision * recall / (precision + recall))
        count = count + 1

    print(
        'average precision: ' + str(np.mean(prec)))
    print('average recall : ' + str(
        np.mean(rec)))
    print('average F score: ' + str(
        np.mean(jaccard_similarity)))

def get_DCG(groundtruth, pred_rank_list,k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1)/math.log2(count+1+1)
        count += 1

    return dcg

def get_NDCG1(groundtruth, pred_rank_list,k):
    count = 0
    dcg = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            dcg += (1)/math.log2(count+1+1)
        count += 1
    idcg = 0
    num_real_item = np.sum(groundtruth)
    num_item = int(num_real_item)
    for i in range(num_item):
        idcg += (1) / math.log2(i + 1 + 1)
    ndcg = dcg / idcg
    return ndcg

def get_HT(groundtruth, pred_rank_list,k):
    count = 0
    for pred in pred_rank_list:
        if count >= k:
            break
        if groundtruth[pred] == 1:
            return 1
        count += 1

    return 0

#input_size = 100
topk = 10

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def merge_history(sum_history_test,test_key_set,training_sum_history_test,training_key_set,index,alpha):
    merged_history = {}
    for test_key_id in range(len(test_key_set)):
        test_key = test_key_set[test_key_id]
        test_history = sum_history_test[test_key]
        sum_training_history = np.zeros(len(test_history))
        for indecis in index[test_key_id]:
            training_key = training_key_set[indecis]
            sum_training_history += training_sum_history_test[training_key]

        sum_training_history = sum_training_history/len(index[test_key_id])

        merge = test_history*alpha + sum_training_history*(1-alpha)
        merged_history[test_key] = merge

    return merged_history

def merge_history_and_neighbors_future(future_data, sum_history_test, test_key_set,training_sum_history_test,
                                       training_key_set,index,alpha, beta):
    merged_history = {}
    for test_key_id in range(len(test_key_set)):
        test_key = test_key_set[test_key_id]
        test_history = sum_history_test[test_key]
        sum_training_history = np.zeros(len(test_history))
        sum_training_future = np.zeros(len(test_history))
        for indecis in index[test_key_id]:
            training_key = training_key_set[indecis]
            sum_training_history += training_sum_history_test[training_key]
            # future_vec = np.zeros((len(test_history)))
            for idx in future_data[training_key][1]:
                if idx >= 0:
                    sum_training_future[idx] += 1

        sum_training_history = sum_training_history/len(index[test_key_id])
        sum_training_future = sum_training_future/len(index[test_key_id])

        merge = (test_history*alpha + sum_training_history*(1-alpha))* beta + sum_training_future*(1-beta)
        merged_history[test_key] = merge

    return merged_history

def evaluate(data_chunk,  training_key_set, test_key_set, input_size, group_size,
             within_decay_rate, group_decay_rate, num_nearest_neighbors, alpha,  topk):
    activate_codes_num = -1
    temporal_decay_sum_history_training = temporal_decay_sum_history(data_chunk[training_chunk],
                                                                     training_key_set, input_size,
                                                                     group_size, within_decay_rate,
                                                                     group_decay_rate)
    temporal_decay_sum_history_test = temporal_decay_sum_history(data_chunk[training_chunk],
                                                                 test_key_set, input_size,
                                                                 group_size, within_decay_rate,
                                                                 group_decay_rate)
    index, distance = KNN(temporal_decay_sum_history_test, temporal_decay_sum_history_training,
                          num_nearest_neighbors)


    sum_history = merge_history(temporal_decay_sum_history_test, test_key_set, temporal_decay_sum_history_training,
                                training_key_set, index, alpha)


    if activate_codes_num < 0:
        # for i in range(1, 6):

        prec = []
        rec = []
        F = []
        prec1 = []
        rec1 = []
        F1 = []
        prec2 = []
        rec2 = []
        F2 = []
        prec3 = []
        rec3 = []
        F3 = []
        NDCG = []
        n_hit = 0

        num_ele = topk
        # print('k = ' + str(activate_codes_num))
        # evaluate(data_chunk, input_size,test_KNN_history, test_key_set, next_k_step)
        count = 0
        for iter in range(len(test_key_set)):
            # training_pair = training_pairs[iter - 1]
            # input_variable = training_pair[0]
            # target_variable = training_pair[1]
            input_variable = data_chunk[training_chunk][test_key_set[iter]]
            target_variable = data_chunk[test_chunk][test_key_set[iter]]

            if len(target_variable) < 2 + next_k_step:
                continue
            count += 1
            output_vectors = predict_with_elements_in_input(sum_history, test_key_set[iter])
            top = 400
            hit = 0
            for idx in range(len(output_vectors)):
                # for idx in [2]:

                output = np.zeros(input_size)
                target_topi = output_vectors[idx].argsort()[::-1][:top]
                c = 0
                for i in range(top):
                    if c >= num_ele:
                        break
                    output[target_topi[i]] = 1
                    c += 1

                vectorized_target = np.zeros(input_size)
                for ii in target_variable[1 + idx]:
                    vectorized_target[ii] = 1
                precision, recall, Fscore, correct = get_precision_recall_Fscore \
                    (vectorized_target, output)
                prec.append(precision)
                rec.append(recall)
                F.append(Fscore)
                if idx == 0:
                    prec1.append(precision)
                    rec1.append(recall)
                    F1.append(Fscore)
                elif idx == 1:
                    prec2.append(precision)
                    rec2.append(recall)
                    F2.append(Fscore)
                elif idx == 2:
                    prec3.append(precision)
                    rec3.append(recall)
                    F3.append(Fscore)
                hit += get_HT(vectorized_target, target_topi, num_ele)
                ndcg = get_NDCG1(vectorized_target, target_topi, num_ele)
                NDCG.append(ndcg)
            if hit == next_k_step:
                n_hit += 1


        # print('average precision of ' + ': ' + str(np.mean(prec)) + ' with std: ' + str(np.std(prec)))
        recall = np.mean(rec)
        ndcg = np.mean(NDCG)
        hr = n_hit / len(test_key_set)

    return recall, ndcg, hr


def main(argv):


    files = [argv[1], argv[2]]

    data_chunk, input_size, code_freq_at_first_claim = read_claim2vector_embedding_file_no_vector(files)

    training_key_set, validation_key_set, test_key_set = partition_the_data_validate(data_chunk, list(data_chunk[test_chunk]), 1)



    num_nearest_neighbors = int(argv[3])
    within_decay_rate = float(argv[4])
    group_decay_rate = float(argv[5])
    alpha = float(argv[6])
    group_size = int(argv[7])
    topk = int(argv[8])

    # num_nearest_neighbors = 300
    # within_decay_rate = 0.9
    # group_decay_rate = 0.7
    # alpha = 0.7
    # group_size = 7
    # topk = 10


    print('Num. of top: ', topk)
    recall, ndcg, hr = evaluate(data_chunk, training_key_set, test_key_set, input_size,
                                group_size, within_decay_rate, group_decay_rate,
                                num_nearest_neighbors, alpha,  topk)

    print('recall: ', str(recall))
    print('NDCG: ', str(ndcg))
    # print('hit ratio: ', str(hr))
    sys.stdout.flush()

if __name__ == '__main__':
    main(sys.argv)