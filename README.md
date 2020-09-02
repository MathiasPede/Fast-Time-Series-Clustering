# Fast Time Series Clustering (ftsc)

Library for faster clustering of large time series datasets. Instead of computing the entire 
distance matrix, this library offers two methods ACA and SOLRADM to approximate this matrix 
using only a limited amount of matrix entries. In fact, the amount of matrix entries necessary
is linear in the amount of time series and the rank of the approximation. Our methods are 
compatible with the distances DTW, ED and also MSM. DTW computations are done using the 
dtaidistance library [github.com/wannesm/dtaidistance](https://github.com/wannesm/dtaidistance). 
The ftsc library also contains a Python and faster C implementation for both ED and MSM similar 
to the DTW implementation of dtaidistance.

## Installation

This library is currently only offered in source code. To offer the full speed of the C version
the following command should be used to compile the Cython files.

    $ python setup.py build_ext --inplace

## Usage

### Loading Time Series
A .tsv file containing a list of time series with the first element of the list the class number,
such as all the datasets on the UCR Time Series Classification Archive[1], should be loaded first.

    from ftsc.data_loader import load_timeseries_from_tsv
    name = "ChlorineConcentration"
    path = "tests/Data/" + name + "/" + name + "_TRAIN.tsv"
    labels, series = load_timeseries_from_tsv(path)

### Creating a ClusterProblem
After loading the dataset, it should be encapsulated in a `ClusterProblem` object, in which also an
empty distance matrix is initiated. This object is used to control which entries of the distance 
matrix are being computed using the compare function provided.

    from ftsc.cluster_problem import ClusterProblem
    
    func_name = "dtw"         # "msm"/"ed" other options
    args = {"window": 100"}   # for MSM "c" parameter
    
    cp = ClusterProblem(series, func_name, compare_args=args)
    
### Approximating the distance matrix
The [DTW/ED/MSM] distance matrix can then be approximated using ACA and SOLRADM.

#### ACA
Adaptive Cross Approximation creates a cross approximation `approx` by adaptively choosing rows of the 
distance matrix. Each of these rows builds a rank 1 matrix and together they form a good approximation
of the full matrix. ACA keeps adding rows until the estimated relative error in the Frobenius norm 
drops below the `tolerance` parameter or after more than `max_rank` rows are chosen.

    from ftsc.aca import aca_symm
    
    approx = aca_symm(cp, tolerance=0.05, max_rank=20)
    
#### SOLRADM
Sample Optimal Low Rank Approximation for Distance Matrices[2] builds an approximation `approx` of the
distance matrix. The algorithm works in 3 steps: first it estimates the 2-norms of the columns of the
matrix. Based on these 2-norms it applies the Monte-Carlo method by randomly selecting an amount of rows
and estimating the column space of the matrix (U matrix in the SVD). Finally in the last step it solves a
minimization problem to compute a matrix V for which UV is a good approximation of the distance matrix.
The approximation has a rank `rank`, which influences together with the "sample factor" `epsilon` (our 
experiments show that `epsilon=2.0` is a good value) the amount of rows that is sampled.

    from ftsc.solradm import solradm
    
    rank = 10
    approx = solradm(cp, rank, epsilon=2.0)

### Full example
This example shows the approximation of DTW distance matrix of the ECG5000 dataset of the UCR Archive. This
dataset contains 5000 time series of length 140.

    from ftsc.data_loader import load_timeseries_from_multiple_tsvs
    from ftsc.cluster_problem import ClusterProblem
    from ftsc.solradm import solradm
    import time

    name = "ECG5000"
    path1 = "Data/" + name + "/" + name + "_TRAIN.tsv"
    path2 = "Data/" + name + "/" + name + "_TEST.tsv"

    labels, series = load_timeseries_from_multiple_tsvs(path1, path2)
    cp = ClusterProblem(series, "dtw")
    
    rank = 50

    start_time = time.time()
    approx = solradm(cp, rank, epsilon=2.0)
    end_time = time.time()
    print("Time spent on approximation: " + str(end_time - start_time) + " seconds")

    start_time = time.time()
    cp.sample_full_matrix()
    end_time = time.time()
    print("Time spent on exact matrix: " + str(end_time - start_time) + " seconds")

    relative_error = cp.get_relative_error(approx)
    print("Relative error of the approximation: " + str(relative_error))

This resulted in:

    Time spent on approximation: 51.12228298187256 seconds
    Time spent on exact matrix: 163.76525115966797 seconds
    Relative error of the approximation: 0.019889041104915007

## References

1. H. A. Dau, E. Keogh, K. Kamgar, C.-C. M. Yeh, Y. Zhu, S. Gharghabi, C. A.Ratanamahatana, Yanping, B. Hu, N. Begum, A. Bagnall, A. Mueen, G. Batista,and Hexagon-ML,
   “The ucr time series classification archive,” October 2018.https://www.cs.ucr.edu/~eamonn/time_series_data_2018/.
2. P. Indyk, A. Vakilian, T. Wagner, and D. P. Woodruff, 
   “Sample-optimal low-rank approximation of distance matrices,” inProceedings of the Thirty-SecondConference on Learning Theory(A. Beygelzimer and D. Hsu, eds.),
   vol. 99 of Proceedings of Machine Learning Research, (Phoenix, USA), pp. 1723–1751,PMLR, 25–28 Jun 2019.


## License

    Copyright 2019-2020 KU Leuven

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
