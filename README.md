# Worm-Tracker

> A toolbox for tracking the position of *C. elegans* in behavioral experiments.



## 0. Prerequisites

### 0.1 Naming Conventions 

Having a consistent and well-defined naming convention can help prevent strange bugs.

- `N2_group_X.avi`: Video recording of a behavioral experiment. `N2` indicates the *C. elegans* strain, `group` refers to the experimental group, and `X` is the plate's i.d..
- `date_correcting.avi`: Video for correction, dated `date`.

### 0.2 **Python Environment**

Create virtual environments for Python through `conda`: 

0. make sure you have installed [anaconda](https://anaconda.com/) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main). 

1. Please ensure you are in the correct directory and execute the following command in your terminal (or CMD/PowerShell on Windows) to create a new environment:
   ```bash
   conda env create -f environment.yml
   ```

   then activate the environment:

   ```bash
   conda activate worm-tracker
   ```

### 0.3 Region of Interest (ROI)

> [!note]
>
> We track worms only within the region of interest (ROI) and discard any trajectories that fall outside of it.

#### 0.3.1 ROI Center

Please extract a frame from the video recording and manually mark the center of the plate with a <span style="color:red; font-weight:bold">red circle</span>, as shown below. 
<img src="./.imgs/ROI_Center.png" style="zoom:25%;" alt="ROI center" />

> [!caution]
>
> Preserve frame resolution during extraction and annotation.

#### 0.3.2 ROI Radius

The parameter of ROI radius (defined as show below) is determined by the input argument during execution of `main.py`. (The default radius is 900 pixel)


<figure style="text-align:center; margin: 0 auto;">
<img src="./.imgs/P2.svg" style="width:80%;" alt="Definition of ROI radius" />
<em>Definition of ROI radius</em>
</figure>



```bash
python main.py --radius 800
```



### 0.4 Correcting



