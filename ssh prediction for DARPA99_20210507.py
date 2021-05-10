## Last Edit : 9.MAY.2021 ##

## test dataset에 대한 evaluation 진행
## test dataset을 evaluation으로 적용
## test dataset은 predicttion model에서 선정
## x_compare에 목적지 IP, 포트 매핑 필요
## 추가 정보 매핑은


import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pydot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import graphviz
from sklearn.tree import export_graphviz
import sys


def loading_dataset(): ## After loading csv file, Pandas Data Frameset Generation ##
    # time.taken   c.ip   response.code  response.type  sc.byte    cs.byte    method URI    cs.host    Destination Port
    # cs_user_agent    sc_filter_result   category   Destination isp    region no_url
    # ratio_trans_receive  count_total_connect    count_connect_IP
    # log_time_taken   log_cs_byte    log_ratio_trans_receive    log_count_connect_IP
    # log_count_total_connect  avg_count_connect  log_avg_count_connect  transmit_speed_BPS
    # log_transmit_speed_BPS   LABEL

    df = pd.read_csv('training_week1.csv', index_col=0)
    df_test = pd.read_csv('test_week3.csv', index_col=0)

    X_training = df[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS","log_count_connect_IP", "log_avg_count_connect", "Business.time"]]
    Y_training = df[["LABEL"]]
    Z_training = df[["log_time_taken"]]
    K_training = df[["no_url"]]
    L_training = df[["log_ratio_trans_receive"]]
    N_training = df[["Destination", "Destination Port", "no_url"]]

    X_test = df_test[["log_count_total_connect", "log_cs_byte", "log_transmit_speed_BPS", "log_count_connect_IP", "log_avg_count_connect", "Business.time"]]
    Y_test = df_test[["LABEL"]]
    Z_test = df_test[["log_time_taken"]]
    K_test = df_test[["no_url"]]
    L_test = df_test[["log_ratio_trans_receive"]]
    N_test = df_test[["Destination", "Destination Port", "no_url"]]
    return(X_training, K_training, L_training, Z_training, Y_training, N_training, X_test, K_test, L_test, Z_test, Y_test, N_test)

## Hybrid Detection Model Function ##
def hybrid_detection(x, y, n_test, y_pred_rf, y_pred_dtree):
    z_temp = x["No_url"]

    a_temp = []
    n = 0
    for i in z_temp:
        if (y_pred_rf[n] == 'ssh'):
            a_temp.append('ssh')
        elif ((y_pred_rf[n] != 'ssh' and i == 1) and y_pred_dtree[n] == 'ssh'):
            a_temp.append('ssh')
        else:
            a_temp.append(y_pred_rf[n])
        n = n + 1

    y_pred_hybrid = pd.DataFrame(a_temp, columns=['Label'])

    ## 동일 IP:Port 에 대한 일치
    y_pred_hybrid["IP:Port"] = n_test["Destination"] + ":" + n_test["Destination Port"].map(str)

    idx_not_ssh = y_pred_hybrid[y_pred_hybrid["Label"] != 'ssh'].index
    temp_ssh = []
    temp_ssh = y_pred_hybrid.drop(idx_not_ssh)
    temp_ssh = temp_ssh.drop_duplicates()
    temp_ssh = temp_ssh.reset_index()  ## 행번호 추가
    del temp_ssh["index"]

    count = len(temp_ssh)
    for i in range(count):
        idx_ssh = y_pred_hybrid[y_pred_hybrid["IP:Port"] == temp_ssh["IP:Port"][i]].index
        y_pred_hybrid["Label"][idx_ssh] = "ssh"

    return metrics.accuracy_score(y, y_pred_hybrid["Label"]), y_pred_hybrid["Label"]


def Decision_TREE_Visual(dtree, d_name):
    ## Decision Tree Visualization ##
    data_feature_names = ['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)']
    export_graphviz(dtree, feature_names=data_feature_names, out_file=d_name+'/'+"DecisionTree.dot",
    #class_names=["Dev_APP", "mobile_APP", "ssh", "web"], impurity=False)
    class_names=["telnet", "ftp", "smtp", "ssh", "web", "Dev_APP", "mobile_APP", ], impurity=False)
    (graph,) = pydot.graph_from_dot_file(d_name+'/'+"DecisionTree.dot", encoding='utf8')
    graph.write_png("DecisionTree.png") #in MAC OS

    with open(d_name+'/'+"DecisionTree.dot") as f:
        dot_graph = f.read()
        g= graphviz.Source(dot_graph)
        g.render(d_name+'/'+"dtree.png", view=False)


def PCA(X_training, K_training, L_training, Z_training, Y_training, N_training, X_test, K_test, L_test, Z_test, Y_test, N_test):
    features_training = X_training.T
    covariance_matrix_training = np.cov(features_training)
    eig_vals_training, eig_vecs_training = np.linalg.eig(covariance_matrix_training)
    dataset_PCA_training = eig_vals_training[0]/sum(eig_vals_training)
    print("Training Dataset PCA:",dataset_PCA_training)

    projected_X_training = X_training.dot(eig_vecs_training.T[0])
    projected_X_test = X_test.dot(eig_vecs_training.T[0])
    projected_X_training = 100*(projected_X_training - min(projected_X_training))/(max(projected_X_training)-min(projected_X_training)) #PC value normalization
    projected_X_test = 100*(projected_X_test - min(projected_X_test))/(max(projected_X_test)-min(projected_X_test)) #PC value normalization

    dataset_training = pd.DataFrame(projected_X_training, columns=['PC'])
    dataset_training['No_url'] = K_training
    dataset_training['Ratio_trans_receive_(Normal)'] = L_training
    dataset_training['Browse_time_(Normal)'] = Z_training
    dataset_training['Label'] = Y_training
    dataset_training[["Destination", "Destination Port"]] = N_training[["Destination", "Destination Port"]]
    dataset_training.to_csv("Result_output/PC of training data.csv", mode='w')

    dataset_test = pd.DataFrame(projected_X_test, columns=['PC'])
    dataset_test['No_url'] = K_test
    dataset_test['Ratio_trans_receive_(Normal)'] = L_test
    dataset_test['Browse_time_(Normal)'] = Z_test
    dataset_test['Label'] = Y_test
    dataset_test[["Destination", "Destination Port"]] = N_test[["Destination", "Destination Port"]]


    ## PCA Graph ##
    sns.set(style="darkgrid") # graph backgroud color
    mpl.rcParams['legend.fontsize'] = 10
    sns.scatterplot('PC', 'Ratio_trans_receive_(Normal)', data=dataset_training, hue="Label", style="No_url", s=40, palette="Set2")
    plt.title("PCA result")
    #plt.show()

    return(dataset_training, dataset_test, dataset_PCA_training)


def SSH_prediction_model(dataset_training, dataset_test, i):
    ## Prediction Modeling ##
    x_training = dataset_training[['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)']]
    y_training = dataset_training[['Label']]
    z_training = dataset_training[['No_url']]
    n_training = dataset_training[["Destination", "Destination Port"]]

    ## Model sampling ##
    x_sample_train, x_sample_test, y_sample_train, y_sample_test, z_sample_train, z_sample_test, n_sample_train, n_sample_test = train_test_split(x_training, y_training, z_training, n_training, test_size= 0.3)

    ## random foreset Modeling ##
    forest100 = RandomForestClassifier(n_estimators=100)
    forest100.fit(x_sample_train,y_sample_train.values.ravel())
    y_sample_test_pred_rf100 = forest100.predict(x_sample_test)
    #x_test에서 ssh 아닌 것 만 발라내기
    print(i,",",'Random Forest Model Accuracy Rate (n=100):',",", metrics.accuracy_score(y_sample_test, y_sample_test_pred_rf100))

    ## Decision Tree Modeling ##
    dtree4 = DecisionTreeClassifier(max_depth=4, random_state=0)
    dtree4.fit(x_sample_train, y_sample_train)
    y_sample_test_pred_dtree4 = dtree4.predict(x_sample_test)
    print(i,",",'Decision Tree Model Accuracy Rate (depth=4):',",", metrics.accuracy_score(y_sample_test, y_sample_test_pred_dtree4))

    ## Sample Test Data Prediction ##
    x_sample_test = x_sample_test.reset_index()
    del x_sample_test["index"]
    y_sample_test = y_sample_test.reset_index()
    del y_sample_test["index"]
    n_sample_test = n_sample_test.reset_index()
    del n_sample_test["index"]

    ## Hybrid Modeling ##
    hybrid_sample_test_accuracy1, y_sample_test_pred_hybrid1 = hybrid_detection(x_sample_test, y_sample_test, n_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree4)
    print(i,",",'Hybrid Model Accuracy Rate (n=100 depth=4):', ",", hybrid_sample_test_accuracy1)


    ## Test(Evaluation) Data Prediction ##
    x_test = dataset_test[['PC', 'No_url', 'Ratio_trans_receive_(Normal)', 'Browse_time_(Normal)']]
    y_test = dataset_test[['Label']]
    z_test = dataset_test[['No_url']]
    N_test = dataset_test[["Destination", "Destination Port"]]

    y_test_predict_rf = forest100.predict(x_test)
    test_label_rf100 = pd.DataFrame(y_test_predict_rf)
    y_test_predict_dt = dtree4.predict(x_test)
    test_label_dt4 = pd.DataFrame(y_test_predict_dt)
    test_hybrid_accuracy1, test_label_hybrid1 = hybrid_detection(x_test, y_test, N_test, y_test_predict_rf, y_test_predict_dt)

    print(i,",",'Test Accuracy Rate of Test DATA(RF):', ",", metrics.accuracy_score(y_test, test_label_rf100))
    print(i,",",'Test Accuracy Rate of Test DATA(DT):', ",", metrics.accuracy_score(y_test, test_label_dt4))
    print(i,",",'Test Accuracy Rate of Test DATA(Hybrid):', ",", metrics.accuracy_score(y_test, test_label_hybrid1))

    return x_test, y_test, z_test, N_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree4, y_sample_test_pred_hybrid1, z_sample_test, n_sample_test, test_label_rf100, test_label_dt4, test_label_hybrid1, test_hybrid_accuracy1, dtree4


def result_pred_output(y_test, z_test, N_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree4, y_sample_test_pred_hybrid1, z_sample_test, n_sample_test, test_label_rf100, test_label_dt4, test_label_hybrid1):

    #Training Dataset
    y_sample_test_pred_total = pd.DataFrame(y_sample_test, columns=["Label"])
    y_sample_test_pred_total['Label_RF100'] = y_sample_test_pred_rf100
    y_sample_test_pred_total['Label_DT4'] = y_sample_test_pred_dtree4
    y_sample_test_pred_total['Label_HYBRID_100_4'] = y_sample_test_pred_hybrid1

    #x_test를 evaluation dataset으로 사용하기 위해 임시 N_compare 생성
    z_sample_test = z_sample_test.reset_index()
    del z_sample_test["index"]
    n_sample_test["No_url"] = z_sample_test

    y_sample_test_pred_total = y_sample_test_pred_total.join(n_sample_test)
    y_sample_test_pred_total.to_csv("Result_output/result_training(sample_test).csv", mode='w')

    y_sample_test_pred_pivot_rf100 = y_sample_test_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_sample_test_pred_pivot_dt4 = y_sample_test_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT4'], aggfunc="count")
    y_sample_test_pred_pivot_hybrid1 = y_sample_test_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_4'], aggfunc="count")

    r0 = ssh_count(y_sample_test_pred_pivot_rf100)  #r1[0] : SSH Detection rate of RF, r1[1] : Total Detection rate of RF, r1[2] : False rate of RF
    d0 = ssh_count(y_sample_test_pred_pivot_dt4)    #d1[0] : SSH Detection rate of DT, d1[1] : Total Detection rate of DT,  d1[2] : False rate of DT
    h0 = ssh_count(y_sample_test_pred_pivot_hybrid1)    #h1[0] : SSH Detection rate of Hybrid, h1[1] : Total Detection rate of Hybrid,  h1[2] : False rate of Hybrid
    print(i,',', "Training/Validation Dataset Total Detection Rate(RF):",',', r0[1])
    print(i,',', "Training/Validation Dataset Total Detection Rate(DT):", ',', d0[1])
    print(i,',', "Training/Validation Dataset Total Detection Rate(Hybrid):",',', h0[1])
    print(i,',', "Training/Validation Dataset SSH Precision(RF):",',', r0[5])
    print(i,',', "Training/Validation Dataset SSH Precision(DT):", ',', d0[5])
    print(i,',', "Training/Validation Dataset SSH Precision(Hybrid):",',', h0[5])
    print(i,',', "Training/Validation Dataset SSH Recall(RF):",',', r0[6])
    print(i,',', "Training/Validation Dataset SSH Recall(DT):", ',', d0[6])
    print(i,',', "Training/Validation Dataset SSH Recall(Hybrid):",',', h0[6])
    print(i,',', "Training/Validation Dataset False Positive Rate of total detection(RF):", ',', r0[3])
    print(i,',', "Training/Validation Dataset False Positive Rate of total detection(DT):", ',', d0[3])
    print(i,',', "Training/Validation Dataset False Positive Rate of total detection(Hybrid):", ',', h0[3])
    print(i,',', "Training/Validation Dataset True Positive Gap of SSH_detection Rate(Hybrid-RF):",',', h0[0]-r0[0]) ## Hybrid? Random Forest? SSH ??? ??

    # Evaluation test Dataset
    y_test_pred_total = pd.DataFrame(y_test, columns=["Label"])
    y_test_pred_total['Label_RF100'] = test_label_rf100
    y_test_pred_total['Label_DT4'] = test_label_dt4
    y_test_pred_total['Label_HYBRID_100_4'] = test_label_hybrid1
    y_test_pred_total = y_test_pred_total.join(z_test)
    y_test_pred_total = y_test_pred_total.join(N_test)
    y_test_pred_total.to_csv("Result_output/result_test.csv", mode='w')

    y_test_pred_pivot_rf100 = y_test_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_test_pred_pivot_dt4 = y_test_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT4'], aggfunc="count")
    y_test_pred_pivot_hybrid1 = y_test_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_4'], aggfunc="count")

    r1 = ssh_count(y_test_pred_pivot_rf100)  #r1[0] : SSH Detection rate of RF, r1[1] : Total Detection rate of RF, r1[2] : False rate of RF
    d1 = ssh_count(y_test_pred_pivot_dt4)    #d1[0] : SSH Detection rate of DT, d1[1] : Total Detection rate of DT,  d1[2] : False rate of DT
    h1 = ssh_count(y_test_pred_pivot_hybrid1)    #h1[0] : SSH Detection rate of Hybrid, h1[1] : Total Detection rate of Hybrid,  h1[2] : False rate of Hybrid
    print(i,',', "Evaluation Test Total Detection Rate(RF):",',', r1[1])
    print(i,',', "Evaluation Test Total Detection Rate(DT):", ',', d1[1])
    print(i,',', "Evaluation Test Total Detection Rate(Hybrid):",',', h1[1])
    print(i,',', "Evaluation Test SSH Precision(RF):",',', r1[5])
    print(i,',', "Evaluation Test SSH Precision(DT):", ',', d1[5])
    print(i,',', "Evaluation Test SSH Precision(Hybrid):",',', h1[5])
    print(i,',', "Evaluation Test SSH Recall(RF):",',', r1[6])
    print(i,',', "Evaluation Test SSH Recall(DT):", ',', d1[6])
    print(i,',', "Evaluation Test SSH Recall(Hybrid):",',', h1[6])
    print(i,',', "Evaluation Test False Positive Rate of total detection(RF):", ',', r1[3])
    print(i,',', "Evaluation Test False Positive Rate of total detection(DT):", ',', d1[3])
    print(i,',', "Evaluation Test False Positive Rate of total detection(Hybrid):", ',', h1[3])
    print(i,',', "Evaluation Test True Positive Gap of SSH_detection Rate(Hybrid-RF):",',', h1[0]-r1[0]) ## Hybrid? Rndom Forest? SSH ??? ??

    return r1, d1, h1

def result_pred_write(y_test, z_test, label_rf100, label_dt4, label_hybrid1, N_test, d_name):
    # Saving the prediction data with cases#
    y_pred_total = pd.DataFrame(y_test, columns=["Label"])
    y_pred_total['Label_RF100'] = label_rf100
    y_pred_total['Label_DT4'] = label_dt4
    y_pred_total['Label_HYBRID_100_4'] = label_hybrid1
    y_pred_total = y_pred_total.join(z_test)
    y_pred_total = y_pred_total.join(N_test)
    y_pred_total.to_csv(d_name+'/'+"row_result_compare.csv", mode='w')


    y_pred_pivot_rf100 = y_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_RF100'], aggfunc="count")
    y_pred_pivot_dt4 = y_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_DT4'], aggfunc="count")
    y_pred_pivot_hybrid1 = y_pred_total.pivot_table('No_url', ['Destination', 'Destination Port', 'Label'], ['Label_HYBRID_100_4'], aggfunc="count")

    y_pred_pivot_rf100.to_csv(d_name+'/'+"pivot_rf100.csv", mode='w')
    y_pred_pivot_dt4.to_csv(d_name+'/'+"pivot_dt4.csv", mode='w')
    y_pred_pivot_hybrid1.to_csv(d_name+'/'+"pivot_hybrid_100_4.csv", mode='w')

def ssh_count(y_pred_pivot): ## SSH Count with IP:Port
    y_pred_pivot.reset_index(level=["Label"], inplace=True)     # Index 속성을 Column 속성으로 변환
    ssh_site = 0
    etc_site = 0
    n = 0
    true_positive_count = 0
    true_negative_count = 0
    false_positive_count = 0
    false_negative_count = 0
    m = len(y_pred_pivot)
    while n < m:
        if (y_pred_pivot["Label"][n] == "ssh"):
            ssh_site = ssh_site + 1
            if (y_pred_pivot["ssh"][n] >= 1):     # True Positive for SSH
                true_positive_count=true_positive_count+1
            else:                                 # False Negative for SSH
                false_negative_count=false_negative_count+1

        if (y_pred_pivot["Label"][n] != "ssh"):
            etc_site = etc_site + 1
            if (y_pred_pivot["ssh"][n] >= 1):      # False Positive for SSH
                false_positive_count=false_positive_count+1
            else:                                  # True Negative for SSH
                true_negative_count = true_negative_count + 1
        n=n+1

    total_detection = (true_positive_count+true_negative_count)/(ssh_site+etc_site)
    false_rate = (false_positive_count+true_negative_count)/(ssh_site+etc_site)
    ssh_detection = true_positive_count/ssh_site          ## true positive detection rate = recall
    false_positive_rate = false_positive_count/etc_site
    true_negative_rate = true_negative_count/etc_site
    precision = true_positive_count / (true_positive_count + false_positive_count)
    recall = true_positive_count / (true_positive_count + false_negative_count)

    return ssh_detection, total_detection, false_rate, false_positive_rate, true_negative_rate, precision, recall



if __name__ == '__main__':
    pd.set_option('display.max_rows', 100)     # Maximum rows for print
    pd.set_option('display.max_columns', 20)   # Maximum columns for print
    pd.set_option('display.width', 20)         # Maximum witch for print
    np.set_printoptions(threshold=100000)
    X_training, K_training, L_training, Z_training, Y_training, N_training, X_test, K_test, L_test, Z_test, Y_test, N_test = loading_dataset()
    dataset_training, dataset_test, dataset_training_PCA = PCA(X_training, K_training, L_training, Z_training, Y_training, N_training, X_test, K_test, L_test, Z_test, Y_test, N_test)
    sys.stdout = open('Result_output/output100.csv', 'w')       # Print as file #
    i=0
    performance_compare1 = 0
    performance_compare2 = 0
    ssh_gap_compare1 = 0.0
    ssh_gap_compare2 = 0.0
    false_compare = 1.0
    sum_SSH_Detection_RF = 0.0
    sum_SSH_Detection_DT = 0.0
    sum_SSH_Detection_HB = 0.0
    sum_total_Detection_RF = 0.0
    sum_total_Detection_DT = 0.0
    sum_total_Detection_HB = 0.0
    sum_false_positive_RF = 0.0
    sum_false_positive_DT = 0.0
    sum_false_positive_HB = 0.0

    print('no', ',', "model", ',', 'rate')
    while i<100:  ## Repeating N times for Predictive model approval
        i=i+1
        #print("This sequence is;",i)
        x_test, y_test, z_test, N_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree4, y_sample_test_pred_hybrid1, z_sample_test, n_sample_test, test_label_rf100, test_label_dt4, test_label_hybrid1, test_hybrid_accuracy1, dtree4 = SSH_prediction_model(dataset_training, dataset_test, i)
        r1, d1, h1 = result_pred_output(y_test, z_test, N_test, y_sample_test, y_sample_test_pred_rf100, y_sample_test_pred_dtree4, y_sample_test_pred_hybrid1, z_sample_test, n_sample_test, test_label_rf100, test_label_dt4, test_label_hybrid1)
        ssh_gap = h1[0] - r1[0]
        RF_DT_gap = abs(d1[0] - r1[0])  # RF - DT 간 편차가 새로운 탐지율에 미치는 영향
        false_rate = h1[2]
        false_positive_rate = h1[3]

        if test_hybrid_accuracy1 >= performance_compare1:   ## total row accuracy  SSH in Hybrid Model
            performance_compare1 = test_hybrid_accuracy1
            case_max_row_rate = i
            d_name = 'Result_Best_Row_Accuracy'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_test, z_test, test_label_rf100, test_label_dt4, test_label_hybrid1, N_test, d_name)
        if h1[0] >= performance_compare2:  ## Largest IP:Port Detection rate in Hybrid Model
            performance_compare2 = h1[0]
            case_max_IP_port_rate = i
            d_name = 'Result_Best_IP_Port_Accuracy'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_test, z_test, test_label_rf100, test_label_dt4, test_label_hybrid1, N_test, d_name)
        if false_positive_rate <= false_compare:  ## Lowest false_positive rate in Hybrid Model
            false_compare = false_positive_rate
            case_minimum_False_rate = i
            d_name = 'Result_Lowest_ssh_False_Positive_rate'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_test, z_test, test_label_rf100, test_label_dt4, test_label_hybrid1, N_test, d_name)
        if ssh_gap >= ssh_gap_compare1:   ## Largest gap of Detection rate between Hybrid and random forest
            ssh_gap_compare1 = ssh_gap
            case_max_gap_Hybrid_RF = i
            d_name = 'Result_Largest_ssh_gap(Hybrid-RF)'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_test, z_test, test_label_rf100, test_label_dt4, test_label_hybrid1, N_test, d_name)
        if RF_DT_gap >= ssh_gap_compare2: ## Largest gap of Detection rate between random forest and decision tree
            ssh_gap_compare2 = RF_DT_gap
            case_max_gap_RF_DT = i
            d_name = 'Result_Largest_ssh_gap(RF-DT)'
            Decision_TREE_Visual(dtree4, d_name)
            result_pred_write(y_test, z_test, test_label_rf100, test_label_dt4, test_label_hybrid1, N_test, d_name)


        sum_SSH_Detection_RF = sum_SSH_Detection_RF + r1[0]
        sum_SSH_Detection_DT = sum_SSH_Detection_DT + d1[0]
        sum_SSH_Detection_HB = sum_SSH_Detection_HB + h1[0]
        sum_total_Detection_RF = sum_total_Detection_RF + r1[1]
        sum_total_Detection_DT = sum_total_Detection_DT + d1[1]
        sum_total_Detection_HB = sum_total_Detection_HB + h1[1]
        sum_false_positive_RF = sum_false_positive_RF + r1[3]
        sum_false_positive_DT = sum_false_positive_DT + d1[3]
        sum_false_positive_HB = sum_false_positive_HB + h1[3]

    false_positive_rate_rf =  sum_false_positive_RF / i
    false_positive_rate_dt =  sum_false_positive_DT / i
    false_positive_rate_hb =  sum_false_positive_HB / i

    sys.stdout = sys.__stdout__ # start of stdout
    sys.stdout = open('Result_output/information.txt', 'w')  # Print as file #
    print("maximum row detection case :", case_max_row_rate)
    print("maximum IP_port detection case :", case_max_IP_port_rate)
    print("minimum False rate detection case :", case_minimum_False_rate)
    print("Detection gap between Decision Tree and Random Forest :", case_max_gap_RF_DT)
    print("Largest detection gap between Hybrid and Random Forest :", case_max_gap_Hybrid_RF)
    print("AVG Total Detection (based IP_port) >>> "
          "\n                                    RandomForest:", sum_total_Detection_RF / i,
          "\n                                    Decision Tree:", sum_total_Detection_DT / i,
          "\n                                    Hybrid:", sum_total_Detection_HB / i)

    print("AVG SSH Detection (based IP_port)   >>> "
          "\n                                    RandomForest:", sum_SSH_Detection_RF / i,
          "\n                                    Decision Tree:", sum_SSH_Detection_DT / i,
          "\n                                    Hybrid:", sum_SSH_Detection_HB / i)

    print("False Positive Rate (based IP_port) >>> "
          "\n                                    RandomForest:", sum_false_positive_RF / i,
          "\n                                    Decision Tree:", sum_false_positive_DT / i,
          "\n                                    Hybrid:", sum_false_positive_HB / i)
    sys.stdout = sys.__stdout__  # End of stdout

