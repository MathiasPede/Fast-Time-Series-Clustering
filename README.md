# Fast Time Series Clustering (ftsc)

Library for faster clustering of time series datasets. Instead of computing the entire 
distance matrix, this library offers two methods ACA and SOLRADM to approximate this matrix 
using only a limited amount of matrix entries. These methods are compatible with the distances
DTW, ED and also MSM. DTW computations are done using the dtaidistance library 
[github.com/wannesm/dtaidistance](https://github.com/wannesm/dtaidistance). The ftsc library also
contains a Python and faster C implementation for both ED and MSM similar to the DTW implementation
of dtaidistance.

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
    
    cp = ClusterProblem(series, "dtw", compare_args=args)
    
### Approximating the distance matrix
The [DTW/ED/MSM] distance matrix can then be approximated using ACA and SOLRADM.

#### ACA
Adaptive Cross Approximation creates a cross approximation by adaptively choosing rows of the 
distance matrix. Each of these rows builds a rank 1 matrix and together they form a good approximation
of the full matrix. ACA keeps adding rows until the estimated relative error in the Frobenius norm 
drops below the `tolerance` parameter or after more than `max_rank` rows are chosen.

    from ftsc.aca import aca_symm
    
    approx = aca_symm(cp, tolerance=0.05, max_rank=20)
    
#### SOLRADM
Sample Optimal Low Rank Approximation for Distance Matrices estimates the 2-norms of the columns of the
matrix and based on that applies the Monte-Carlo method by randomly selecting an amount of rows. This 
amount depends on the desired `rank` of the approximation and the "sample factor" `epsilon`.

    from ftsc.solradm import solradm
    
    rank = 10
    approx = solradm(cp, rank, epsilon=2.0)


## References

1. C. Yanping, K. Eamonn, H. Bing, B. Nurjahan, B. Anthony, M. Abdullah and B. Gustavo.
   [The UCR Time Series Classification Archive](www.cs.ucr.edu/~eamonn/time_series_data/), 2015.



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
