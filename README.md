# FCN-rLSTM model trained on webcamt dataset
Capstone project : Density map estimation &amp; Vehicle Counting
<hr/>

## Datasets
* WebCamT 
    - https://github.com/Lotuslisa/WebCamT (Github)
    - https://www.citycam-cmu.com/ (Dataset url)

## Model
* FCN-rLSTM
    - [Shanghang Zhang, Guanhang Wu, Joao P Costeira, and Jose MF Moura. Fcn-rlstm: Deep spatio-temporal neural networks for vehicle counting in city cameras. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3667–3676, 2017.](https://arxiv.org/abs/1707.09476)
    - [Implementation notes - dpernes/FCN-rLSTM](https://github.com/dpernes/FCN-rLSTM)

## Details
> Memory overload occurs when about 59000 pieces of data are loaded at once

* Save the preprocessed data as a .pickle file at once or divided into two depending on the capacity of each folder.
* For each epoch, all pickles are loaded and trained, and after training on the last pickle file, the training results are saved to a log file in the path set for checking the results.
* Training and test data can be modified as needed.

## Contact
* hongil7626@gmail.com