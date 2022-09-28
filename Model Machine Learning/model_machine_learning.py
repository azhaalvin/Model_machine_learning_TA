from tkinter import *
from tkinter.ttk import Combobox

 
window=Tk()
var = StringVar()
var.set("one")
#data=("one", "two", "three", "four")
#cb=Combobox(window, values=data)
#cb.place(x=60, y=150)

#lb=Listbox(window, height=5, selectmode='multiple')
#for num in data:
#    lb.insert(END,num)
#lb.place(x=250, y=150)


class Table:
      
    def __init__(self,root,rows):
        # code for creating table
        newWindow = Toplevel(window)
        newWindow.title("Data Test")
        # sets the geometry of toplevel
        newWindow.geometry("600x400")

        for i in range(len(rows)):
            for j in range(10):
                self.e=Entry(newWindow)
                  
                self.e.grid(row=i, column=j)
                self.e.insert(END, rows[i][j])

         



#handle tombol klik
def tombol_klik():
    tombol["state"] = DISABLED
    if (v0.get()<2):
        metoda=1
    else:
        if (v0.get()==2):
            metoda=2
        else:
            metoda=3
    print(metoda)
    V=[]
    if (v1.get()==1):
        if not V:
            V=[1]
        else:
            V.append(1)
    if (v2.get()==1):
        if not V:
            V=[2]
        else:
            V.append(2)
    if (v3.get()==1):
        if not V:
            V=[3]
        else:
            V.append(3)
    if (v4.get()==1):
        if not V:
            V=[4]
        else:
            V.append(4)
    if (v5.get()==1):
        if not V:
            V=[5]
        else:
            V.append(5)
    if (v6.get()==1):
        if not V:
            V=[6]
        else:
            V.append(6)
    if (v7.get()==1):
        if not V:
            V=[7]
        else:
            V.append(7)
    if not V:
        tombol["state"] = NORMAL
        print('Fitur tidak boleh kosong, Selesai ...')
        return
    print(V)
    from scipy.signal import kaiserord, lfilter, firwin, freqz
    import scipy as sp
    import numpy as np
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import signal
    from numpy import savetxt
    from scipy.stats import entropy
    from math import log, e
    import os, sys
    from scipy.signal import butter, iirnotch, lfilter
    from scipy.stats import norm, kurtosis
    from scipy.stats import skew
    ## A high pass filter allows frequencies higher than a cut-off value
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False, output='ba')
        return b, a
    ## A low pass filter allows frequencies lower than a cut-off value
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5*fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
        return b, a
    def notch_filter(cutoff, q):
        nyq = 0.5*fs
        freq = cutoff/nyq
        b, a = iirnotch(freq, q)
        return b, a

    def final_filter(data, fs, order=5):
        b, a = butter_highpass(cutoff_high, fs, order=order)
        x = lfilter(b, a, data)
        d, c = butter_lowpass(cutoff_low, fs, order = order)
        y = lfilter(d, c, x)
        f, e = notch_filter(powerline, 30)
        z = lfilter(f, e, y)     
        return z

    def Mean(data):
        """Returns the mean of a time series"""
        return data.mean()

    def Std(data):
        """Returns the standard deviation a time series"""
        return data.std()

    # def Min(data):
    #     """Returns the mean of a time series"""
    #     return data.min()

    def Max(data):
        """Returns the standard deviation a time series"""
        return data.max()

    def entropy1(labels, base=None):
       value,counts = np.unique(labels, return_counts=True)
       return entropy(counts, base=base)

    J=0 # jumlah file
    directory_path = 'dataset/sehat'
    for iter in range(0,2):
        for x in os.listdir(directory_path):
            if not x.lower().endswith('.csv'):
                continue
            J=J+1
        directory_path = 'dataset/pasien'
    n = J #jumlah file
    m = 8
    FEAT = [] #bakal jadi Feature.csv
    for i in range(n): 
        FEAT.append([0] * m) #mengisi dengan angka 0 semua
 
    directory_path = 'dataset/sehat'
    J=-1
    K=0
    for iter in range(0,2):
        for x in os.listdir(directory_path):
            if not x.lower().endswith('.csv'):
                continue
            full_file_path = directory_path  +   '/'   + x
            J=J+1
            print ('Using file', full_file_path)
            try:
                dataraw = pd.read_csv(full_file_path,index_col='Timestamp', parse_dates=['Timestamp'])
                dataset = pd.DataFrame(dataraw['Value']) #ambil kolom value dari setiap file
            except:
                dataraw = pd.read_csv(full_file_path,index_col='timestamp', parse_dates=['timestamp'])
                dataset = pd.DataFrame(dataraw['values'])
            x1=np.array(dataset)  #ubah jadi array, namanya x1
            Dat=[]
            Dat=[0 for i in range(x1.size)] #bikin array kosong isinya 0 semua sepajang array x1
            for i in range(0,x1.size-1):
                Dat[i]=max(x1[i])
                
            # FIR filter
            cutoff_low = 2
            powerline = 60
            fs = 1000
            nyq_rate = fs / 2.0
            width = 5.0/nyq_rate
            N, beta = kaiserord(powerline, width)
            cutoff_hz = cutoff_low
            taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
            y3_filtered = lfilter(taps, 1.0, Dat)
            if (metoda==3):
                # FIR Filter
                y_filtered=y3_filtered

            import collections
            import neurokit2 as nk

            def shannon_entropy(data):
                bases = collections.Counter([tmp_base for tmp_base in data])
                # define distribution
                dist = [x / sum(bases.values()) for x in bases.values()]

                # use scipy to calculate entropy
                entropy_value = entropy(dist, base=2)

                return entropy_value

            # FEATURE EXTRACTION
            try:
                # Shannon entropy
                ESH = shannon_entropy(y_filtered)
                FEAT[J][0] = ESH

                # MAD
                series = pd.Series(y_filtered)
                MAD = series.mad()
                # Kurtosis
                KURT = kurtosis(y_filtered)
                # Skewness
                SKEW = skew(y_filtered)
                # vlf,lf,hf
                info = nk.ppg_findpeaks(y_filtered)
                peak = info["PPG_Peaks"]
                hrv_freq = nk.hrv_frequency(peak, sampling_rate=39, normalize=True)
                VLF = hrv_freq['HRV_VLF'].values[0]
                LF = hrv_freq['HRV_LF'].values[0]
                HF = hrv_freq['HRV_HF'].values[0]
                FEAT[J][0] = ESH
                FEAT[J][1] = MAD
                FEAT[J][2] = KURT
                FEAT[J][3] = SKEW
                FEAT[J][4] = VLF
                FEAT[J][5] = LF
                FEAT[J][6] = HF
                FEAT[J][7] = K
            except:
                J = J - 1
        directory_path = 'dataset/pasien'
        K=1
    
    #sehat = 0, pasien = 1

    #building data uji
    J = 1
    m = 10
    directory_path = 'dataset/uji'
    for x in os.listdir(directory_path):
        if not x.lower().endswith('.csv'):
            continue
        J=J+1
    n = J
    FEAT2 = [] #bakal jadi Feature.csv
    for i in range(n): 
        FEAT2.append([0] * m) #mengisi dengan angka 0 semua
    J=-1
    for x in os.listdir(directory_path):
        if not x.lower().endswith('.csv'):
            continue
        full_file_path = directory_path  +   '/'   + x
        J=J+1
        print ('Using file', full_file_path)
        try:
            dataraw = pd.read_csv(full_file_path,index_col='Timestamp', parse_dates=['Timestamp'])
            dataset = pd.DataFrame(dataraw['Value']) #ambil kolom value dari setiap file
        except:
            dataraw = pd.read_csv(full_file_path,index_col='timestamp', parse_dates=['timestamp'])
            dataset = pd.DataFrame(dataraw['values'])
        x1=np.array(dataset)  #ubah jadi array, namanya x1
        Dat=[]
        Dat=[0 for i in range(x1.size)] #bikin array kosong isinya 0 semua sepajang array x1
        for i in range(0,x1.size-1):
            Dat[i]=max(x1[i]) #why pakai max?
            
        # FIR filter
        fs = 1000
        cutoff_low = 2
        nyq_rate = fs / 2.0
        width = 5.0/nyq_rate
        # The desired attenuation in the stop band, in dB.
        # Compute the order and Kaiser parameter for the FIR filter.
        N, beta = kaiserord(powerline, width)
        cutoff_hz = cutoff_low
        taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
        y3_filtered = lfilter(taps, 1.0, Dat)
        if (metoda==3):
            # FIR Filter
            y_filtered=y3_filtered

        # FEATURE EXTRACTION
        try:
            # Shannon entropy
            ESH = shannon_entropy(y_filtered)
            FEAT2[J][0] = ESH

            # MAD
            series = pd.Series(y_filtered)
            MAD = series.mad()
            # Kurtosis
            KURT = kurtosis(y_filtered)
            # Skewness
            SKEW = skew(y_filtered)

            # vlf,lf,hf
            info = nk.ppg_findpeaks(y_filtered)
            peak = info["PPG_Peaks"]
            hrv_freq = nk.hrv_frequency(peak, sampling_rate=39, normalize=True)
            VLF = hrv_freq['HRV_VLF'].values[0]
            LF = hrv_freq['HRV_LF'].values[0]
            HF = hrv_freq['HRV_HF'].values[0]
            FEAT2[J][0] = ESH
            FEAT2[J][1] = MAD
            FEAT2[J][2] = KURT
            FEAT2[J][3] = SKEW
            FEAT2[J][4] = VLF
            FEAT2[J][5] = LF
            FEAT2[J][6] = HF
            FEAT2[J][7] = K
        except:
            J = J - 1
    import csv
    #bikin csv
    with open("Feature.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(FEAT)

    with open("Feature2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(FEAT2)

    # MACHINE LEARNING
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    # import sklearn
    

    dataset = pd.read_csv('FEATURE.csv', names=['ESH', 'MAD', 'SKEW','KURT','VLF','LF','HF','Label'])
    x = np.random.randint(9, size=(3, 3))
    value = 69
    dataset['VLF'] = dataset['VLF'].fillna(dataset['VLF'].mean())
    print(dataset)


    # FEATURE SELECTION
    V = [q-1 for q in V]
    print('V = ',V)
    # Menggunakan semua feature
    X = dataset.iloc[:, V].values
    print('value = ',X)

    # Menggunakan feature nomor 1, 2, 3, 4, 5, boleh loncat angkanya mis : 1, 3, 5, 7
    #X = dataset.iloc[:, [1, 2, 3, 4, 5, 9]].values
    y = dataset.iloc[:, -1].values

    # SPLIT DATA 80% TRAIN, 20% DATA TEST
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = value)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    #rows=[['Stock 1', 'Stock 2', 'Value'], ['AXISBANK.NS', 'MAHABANK.NS', 81.10000000000001], ['AXISBANK.NS', 'BANKINDIA.NS', 82.3], ['BANKBARODA.NS', 'MAHABANK.NS', 84.8], ['MAHABANK.NS', 'CANBK.NS', 85.5], ['BANKBARODA.NS', 'BANKINDIA.NS', 90.4], ['BANKINDIA.NS', 'CANBK.NS', 90.9], ['AXISBANK.NS', 'CANBK.NS', 91.5], ['AXISBANK.NS', 'BANKBARODA.NS', 93.30000000000001], ['BANKINDIA.NS', 'MAHABANK.NS', 95.8], ['BANKBARODA.NS', 'CANBK.NS', 97.6]]
    n = len(y_test)
    m = 10
    Data_test = []
    for i in range(n):
        Data_test.append([0] * m)
    for i in range(n):
        a=X_test[i]
        for j in range(len(a)):
            Data_test[i][j]=a[j]

    data_uji = pd.read_csv('FEATURE2.csv')

    def hitung_akurasi(tp,tn,fp,fn):
        acc = float
        acc = (tp+tn)/(tp+tn+fp+fn)
        return acc
    def hitung_sensitifiti(tp, fn):
        sens = float
        sens = tp/(tp+fn)
        return sens
    def hitung_spesifisiti(tn,fp):
        spes = float
        spes = tn/(tn+fp)
        return spes

    # yang bisa dimainin: jumlah n_neighbors, metode,
    # sebelum tunning
    # PROSES KLASIFIKASI MACHINE LEARNING
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import metrics
    classifier = KNeighborsClassifier()

    # mean_acc = np.zeros(30)
    # for i in range(1, 31):
    #     # Train Model and Predict
    #     knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    #     yhat = knn.predict(X_test)
    #     mean_acc[i - 1] = metrics.accuracy_score(y_test, yhat)
    #
    # print('mean acc = ',mean_acc)
    #
    # loc = np.arange(1, 31, step=1.0)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, 31), mean_acc)
    # plt.xticks(loc)
    # plt.xlabel('Number of Neighbors ')
    # plt.ylabel('Accuracy')
    # plt.show()

    from sklearn.model_selection import GridSearchCV

    classifier = KNeighborsClassifier()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # print('HASIL PREDIKSI KNN K-')
    # print(y_pred)

    from sklearn.metrics import confusion_matrix, accuracy_score

    ac = accuracy_score(y_test, y_pred)
    if (metoda == 1):
        # Savitzky-Golay filter
        print('Denoising Savitzky-Golay')
    if (metoda == 2):
        # Butterworth filter
        print('Denoising Butterworth')
    if (metoda == 3):
        # FIR Filter
        print('Denoising FIR')

    knn = ac

    print("Akurasi KNN-", ac * 100, ' %')
    X_uji = data_uji.iloc[:, V]
    predicted = classifier.predict(X_uji)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    acc_knn = hitung_akurasi(tp, tn, fp, fn)
    sens_knn = hitung_sensitifiti(tp, fn)
    spes_knn = hitung_spesifisiti(tn, fp)


    # PROSES KLASIFIKASI MACHINE LEARNING decision tree
    from sklearn import tree
    # melakukan tuning DT start#
    from sklearn.model_selection import GridSearchCV
    # grid_params = {'max_depth': [5 ,10, 15, 20, 50, 100, 120, 140]}
    #
    # gs = GridSearchCV(tree.DecisionTreeClassifier(), grid_params, verbose=2, cv=3, n_jobs=-1)
    # g_res = gs.fit(X_train, y_train)
    #
    # print('gres =', g_res.best_score_)
    # print('best parrrams =', g_res.best_params_)
    # melakukan tuning DT End#

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('HASIL PREDIKSI')
    print(y_pred)

    X_uji = data_uji.iloc[:, V]
    predicted = classifier.predict(X_uji)
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    acc_dt = hitung_akurasi(tp, tn, fp, fn)
    sens_dt = hitung_sensitifiti(tp, fn)
    spes_dt = hitung_spesifisiti(tn, fp)

    from sklearn.metrics import confusion_matrix, accuracy_score

    ac = accuracy_score(y_test, y_pred)
    if (metoda == 1):
        # Savitzky-Golay filter
        print('Denoising Savitzky-Golay')
    if (metoda == 2):
        # Butterworth filter
        print('Denoising Butterworth')
    if (metoda == 3):
        # FIR Filter
        print('Denoising FIR')


    dt = ac
    print("Akurasi Decision Tree", ac * 100, ' %')

    # PROSES KLASIFIKASI MACHINE LEARNING SVM
    from sklearn import svm
    #melakukan tuning SVM start#
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn import decomposition

    from sklearn.preprocessing import LabelEncoder

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                         'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                        {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                        ]

    # tuned_parameters2 = [{'kernel': ['rbf','linear','sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
    #                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
    #                     ]
    #
    # scores = ['precision', 'recall']
    # for score in scores:
    #     print("# Tuning hyper-parameters for %s" % score)
    #     print()
    #
    #     clf = GridSearchCV(SVC(), tuned_parameters2, cv=5,
    #                        scoring='%s_macro' % score)
    #     clf.fit(X_train, y_train)
    #
    #     print("Best parameters set found on development set:")
    #     print()
    #     print(clf.best_params_)
    #     print()
    #     print("Grid scores on development set:")
    #     print()
    #     means = clf.cv_results_['mean_test_score']
    #     stds = clf.cv_results_['std_test_score']
    #     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #         print("%0.3f (+/-%0.03f) for %r"
    #               % (mean, std * 2, params))
    #     print()
    #
    # print(clf.best_params_)
    # melakukan tuning SVM end#

    classifier = svm.SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print('HASIL PREDIKSI')
    print(y_pred)

    X_uji = data_uji.iloc[:, V]
    cm = confusion_matrix(y_test, y_pred)
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]

    acc_svm = hitung_akurasi(tp, tn, fp, fn)
    sens_svm = hitung_sensitifiti(tp, fn)
    spes_svm = hitung_spesifisiti(tn, fp)

    from sklearn.metrics import confusion_matrix,accuracy_score
    cm = confusion_matrix(y_test, y_pred)
    ac = accuracy_score(y_test,y_pred)
    if (metoda==3):
        # FIR Filter
        print('Denoising FIR')

    print("--------------------------------------")
    print ("Akurasi svm", ac*100,' %')
    print("akurasi metrik uji : ", acc_svm)
    print("sensitiviti metrik uji : ", sens_svm)
    print("spesifisiti metrik uji : ", spes_svm)

    print('--------------- ...')
    print("Akurasi Decision Tree", dt * 100, ' %')
    print("akurasi metrik uji : ", acc_dt)
    print("sensitiviti metrik uji : ", sens_dt)
    print("spesifisiti metrik uji : ", spes_dt)

    print('--------------- ...')
    print("Akurasi KNN-", knn * 100, ' %')
    print("akurasi metrik uji : ", acc_knn)
    print("sensitiviti metrik uji : ", sens_knn)
    print("spesifisiti metrik uji : ", spes_knn)

    # print("----------hasil tunning--------")
    # print("hasil tuning svm : ", clf.best_params_)
    # print("hasil tuning dt : ", g_res.best_params_)
    # print("hasil tuning knn : ", )

    tombol["state"] = NORMAL

    print('Selesai ...')
    print('--------------- ...')

var1 = StringVar()
label1 = Label(window, textvariable=var1, relief=RAISED ,width = 107,bg ='cyan'  )

var1.set("Metoda denoding : ")
label1.place(x=20, y=25)
v0=IntVar()
v0.set(1)
r1=Radiobutton(window, text="FIR", variable=v0,value=3)

r1.place(x=20,y=50)

var2 = StringVar()
label2 = Label(window, textvariable=var2, relief=RAISED ,width = 107,bg ='cyan'  )

var2.set("Fitur yang dipilih : ")
label2.place(x=20, y=100)
 
v1 = IntVar()
v2 = IntVar()
v3 = IntVar()
v4 = IntVar()
v5 = IntVar()
v6 = IntVar()
v7 = IntVar()
C1 = Checkbutton(window, text = "Shannon Entropy", variable = v1)
C2 = Checkbutton(window, text = "MAD", variable = v2)
C3 = Checkbutton(window, text = "Kurtosis", variable = v3)
C4 = Checkbutton(window, text = "Skewness", variable = v4)
C5 = Checkbutton(window, text = "VLF", variable = v5)
C6 = Checkbutton(window, text = "LF", variable = v6)
C7 = Checkbutton(window, text = "HF ", variable = v7)
C1.place(x=20, y=125)
C2.place(x=180, y=125)
C3.place(x=340, y=125)
C4.place(x=500, y=125)
C5.place(x=660, y=125)
C6.place(x=20, y=150)
C7.place(x=180, y=150)

tombol = Button(window,
                   text="RUN Klasifikasi",
                   command=tombol_klik,bg='blue',fg='white')
tombol.place(x=350, y=200)

window.title('Klasifikasi PPG Machine Learning')
window.geometry("800x300+10+10")
window.mainloop()

