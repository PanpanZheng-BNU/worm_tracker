# Worm-Tracker

>[!Tip]
> A toolbox for tracking the position of *C. elegans* in behavioral experiments.



## 0. Prerequisites

1. **Naming Conventions**: Having a consistent and well-defined naming convention can help prevent strange bugs.
   - `N2_group_X.avi`: Video recording of a behavioral experiment. `N2` indicates the *C. elegans* strain, `group` refers to the experimental group, and `X` is the plate's i.d..
   - `date_correcting.avi`: Video for correction, dated `date`.
2. **Python Environment**: Create virtual environments for Python through `conda`, make sure you have installed [anaconda](https://anaconda.com/) or [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main). Please ensure you are in the correct directory and execute the following command in your terminal (or CMD/PowerShell on Windows) to create a new environment:
   ```bash
   conda env create -f environment.yml
   ```
   then activate the environment:
   ```bash
   conda activate worm-tracker
    ```
3. **
