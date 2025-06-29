## Dataset Description

### Koopman operator matrix (`Koopman_matrix/`)

- `CAFUC-abnormal.npy`, `CAFUC-abnormal2.npy`, `CAFUC-abnormal3.npy`
The operator matrix of the CAFUC2 dataset with real abnormalities.
- `RectangeDatasets-0.npy`
The real normal operator matrix of the CAFUC2 dataset.
- `RectangeDatasets-0-1000.npy`
Artificial abnormal intervention is performed on the first 0–1000 rows, and the abnormal value range is `(-1, 1)`.
- `RectangeDatasets-0-1000-(-4,4).npy`
Artificial abnormal intervention is performed on the first 0–1000 rows, and the abnormal value range is `(-4, 4)`.
- `RosslerDatasets.npy`
Rossler system normal data set operator matrix.
- `RosslerDatasets-1000.npy`
Rossler data set first 0–1000 rows artificial abnormal intervention, abnormal value range `(-1, 1)`.
- `LorenzAnomDatasets1-2.npy`
Lorenz system normal data set operator matrix.
- `LorenzAnmoDatasets-1000.npy`
Lorenz data set first 0–1000 rows artificial abnormal intervention, abnormal value range `(-1, 1)`.
- `EGG-normal.npy`
EGG system normal data set operator matrix.
- `EGG-abnormal-0-1000.npy`
EGG data set first 0–1000 rows artificial abnormal intervention, abnormal value range `(-1, 1)`.
- `EGG-abnormal-real.npy`
EGG system real abnormal data operator matrix.

### Original time series data (root directory)

- `CAFUC_normal.npy`
CAFUC2 system original time series data without Koopman processing (normal).
- `CAFUC_anomaly-0-1000.npy`
CAFUC2 system first 0–1000 lines of artificial abnormal intervention of original time series data, abnormal value range `(-1, 1)`.
- `EGG_normal.npy`
EGG system original time series data without Koopman processing (normal).
- `EGG_anomaly-0-1000.npy`
EGG system first 0–1000 lines of artificial abnormal intervention of original time series data, abnormal value range `(-1, 1)`.
- `Lorenz_normal.npy`
Raw time series data of the Lorenz system without Koopman processing (normal).
- `Lorenz_anomaly-0-1000.npy`
Raw time series data of the Lorenz system with artificial anomaly intervention for the first 0–1000 rows, with anomaly range of `(-1, 1)`.
- `Rossler_normal.npy`
Raw time series data of the Rossler system without Koopman processing (normal).
- `Rossler_anomaly-0-1000.npy`
Raw time series data of the Rossler system with artificial anomaly intervention for the first 0–1000 rows, with anomaly range of `(-1, 1)`.

---

**Run**: Execute `demo.py` to generate all analysis charts and console indicator outputs in one click. Execute ' ReadME RunME Analysis Forecast.m` to generate modal diagrams.
