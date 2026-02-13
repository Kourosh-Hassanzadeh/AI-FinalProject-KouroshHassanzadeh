"# AI-FinalProject-KouroshHassanzadeh" 


# Environment Setup

Windows:

```
python -m venv env

env\Scripts\activate

pip install -r requirements.txt
```
Linux:

```
python3 -m venv env

source env/bin/activate

pip install -r requirements.txt
```

# EDA Analysis

```
python utils\eda\eda.py
```

The outputs will be saved beside the script.

# Data

You can find the datasets in the `data` directory.  
The data used for each problem is different.

All data:

![alt text](utils/eda/outputs/samples_overall.png)


# Spatial and Frequency Filters

Applying Average, gaussian, median, sharpening, sobel in spatial domain and High Pass and Low Pass in frequency domain.

```
python src\spatial_frequency_filters.py
```

results:
![resulters.pngs/filt](results/filters.png)

# Denoising

Applying Salt&Pepper and gaussian noise and remove them with median and bilateral filters.

```
python src\denoising.py
```

results:

s&p + median:

![results/sap_median.png](results/sap_median.png)

gaussian + bilateral:

![results/gaussian_bilateral.png](results/gaussian_bilateral.png)

# Edge Detection

Applying prewitt, kirsch, Marr_Hildreth, canny Algorithms.

```
python src\edge_detection.py
```

results:

prewitt:

![results/prewitt.png](results/prewitt.png)

kirsch:

![results/kirsch.png](results/kirsch.png)

Marr_Hildreth:

![alt text](results/Marr_Hildreth.png)

canny:

![results/canny.png](results/canny.png)


# Segmentation

Applying Kmeans at pixel level with different k

```
python src\segmentation.py
```

k = 8:

![results/kmeans_8.png](results/kmeans_8.png)

k = 16:

![results/kmeans_16.png](results/kmeans_16.png)

k = 32:

![results/kmeans_32.png](results/kmeans_32.png)