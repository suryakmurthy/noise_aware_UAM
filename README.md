# ILASMS_func3a

This is the git repository for Function 3a of the NASA System-Wide Safety program.


# Installation Mac M1/2

## 1. Install Miniforge


* `wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh`

* `bash Miniforge3-MacOSX-arm64.sh`

* `rm https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh`


## 2. Install project dependencies

1. Close and open a new terminal
2. Navigate to ILSMS_FUNC3A directory
    ```bash
    cd to/ILASMS_FUNC3A
    ```
3. Navigate to the setup directory 
    ```bash
    cd setup
    ```
4. Install dependencies with conda
    ```bash
    conda env create -f environment_mac_silicon.yml
    ```
5. Close and open a new terminal
6. Activate virtual environment
    ```bash
    conda env create -f environment_mac_silicon.yml
    ```
7. Naviagate to project directory (see step 1)
8. Naviagate to BlueSky directory
    ```bash
    cd bluesky
    ```
9. Install bluesky
    ```python
    pip install -e .
    ```

# Running Project

1. Open terminal and activate virtual environment
    ```bash
    conda activate sws
    ```
2. Navigate to ILSMS_FUNC3A directory
    ```bash
    cd to/ILASMS_FUNC3A
    ```
3. Try running main script
    ```python
    python main.py
    ````


# Visualization

1. Follow steps 1-2 above (Section: Running Project) in a single terminal (Terminal 1). Open a **second** terminal (Terminal 2) and follow the steps below
2. Activate virtual environment
    ```bash
    conda activate sws
    ```
2. Navigate to ILSMS_FUNC3A directory
    ```bash
    cd to/ILASMS_FUNC3A
    ```
3. Naviagate to BlueSky directory
    ```bash
    cd bluesky
    ```
4. Start BlueSky
    ```bash
    python BlueSky.py
    ```
5. The GUI should open up. After the GUI has started, in Terminal 1, run step 3 of **Running Project** to start the simulation.
6. In the BlueSky GUI, select the **Nodes** tab on the lower-right side. Select a different simulation node to see the DFW sim.


# Issues

## 1. grpcio error during `python main.py`

If this error is encountered, uninstall grpcio and reinstall with conda

```bash
pip uninstall grpcio; conda install grpcio=1.43.0
```


# Generate multiple scenarios

Run `scripts/multiple_scn_script.py`. The generated scenarios are stored in `scripts/generated_scenarios`.