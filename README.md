# Model_machine_learning_TA
Terdapat 2 program yang ada dalam repositori ini :

1. model machine learning : berisikan tahapan denoising, ekstraksi ciri, dan klasifikasi. dataset yang digunakan adalah sinyal PPG yang berisikian subyek sehat dan subyek pasien PJK. Terdapat 3 klasifikasi yang bisa gunakan yaitu KNN, SVM, dan Decision Tree yang nantinya dapat melakuakn klasifikasi.

2. Program seleksi fitur/ciri : berisikan code khusus untuk memilih fitur yang akan di gunakan. Metode yang dipilih dalam melakukan seleksi fitur ada Pearson Correlation. dataset yang di pakai adalah fitur yang didapat dari metode denoising FIR


Cara menggunakan model machine learning :

1. Download file program
2. Buka file model_machine_learning.py menggunakan texteditor (Penulis menggunakan Pycharm)
3. Klasifikasi yang berada di program ini secara default belum di lakukan tuning, klasifikasi tersebut berada pada baris :
KNN = 418
Decision Tree = baris 468
SVM = baris 546

4. jika ingin melakukan tuning ikuti langkah berikut :
 a) KNN : buka code baris 401-416 dan hapus "comment" atau tanda "#" yang berada pada awal code
 b) Decision Tree : buka code baris 456-466 dan hapus "comment" atau tanda "#" yang berada pada awal code. parameter yang akan di ubah adalah "max_depth", pada variabel grid_params yang berada pada baris 456 dapat diubah sesuai kebutuhan sehingga parameter yang akan di gunakan lebih bervariasi
 c)SVM :buka code baris 513-544 dan hapus "comment" atau tanda "#" yang berada pada awal code. Ada beberpa parameter dari svm ini yang akan di lakukan tuning itu terdiri dari paramter "kernel", "gamma" dan "C", nilai dari parameter tersebut dapat diubah sesuai kebutuhan pada variabel 'tuned_parameters' pada baris 513

5. Lalu run program

6. Jika pop up gui dari program sudah muncul pilih fitur yang akan di gunakan. Ada 7 fitur yang dipakai yaitu MAD, ESH, SKEW, KURT, HF, LF, VLF. Silahkan dipilih 1 atau lebih fitur yang ingin digunakan.

7. Klik tombol "Run Klasifikasi"

8. Hasil performa akan terlihat pada console log texteditor yang di pilih


cara menggunakan program seleksi fitur 

1.Download folder Program Metode Seleksi Fitur

2. Buka file Algoritma_Metode_Seleksi_Fitur.ipynb menggunakan editor Jupyter atau Google Colab

3. Masukan dataset Feature_FIR.csv yang terdapat pada folder dataset
Run Program

4. Gunakan fitur terpilih pada program Model Deteksi apabila ingin melakukan pengujian berdasarkan metode seleksi fitur Pearson Correlation
