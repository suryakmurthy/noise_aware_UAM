# Noise Aware Behavior in UAM Systems

This is the github repository for the final project of ECE 381V: Reinforcement Learning: Theory and Practice

# Installation Ubunutu


## 1. Install project dependencies

1. Close and open a new terminal
2. Navigate to noise_aware_UAM directory
    ```bash
    cd noise_aware_UAM
    ```
3. Close and open a new terminal
4. Naviagate to project directory (see step 1)
5. Install bluesky
    ```python
    pip install -e .
    ```

For more information on the BlueSky Simulator, please see: https://github.com/TUDelft-CNS-ATM/bluesky

# Running Project

1. Navigate to noise_aware_UAM directory
    ```bash
    cd noise_aware_UAM
    ```
2. Try running main script
    ```python
    python main.py
    ````


# Visualization

1. Follow steps 1-2 above (Section: Running Project) in a single terminal (Terminal 1). Open a **second** terminal (Terminal 2) and follow the steps below

2. Navigate to noise_aware_UAM directory
    ```bash
    cd noise_aware_UAM
    ```
3. Start BlueSky
    ```bash
    python BlueSky.py
    ```
4. The GUI should open up. After the GUI has started, in Terminal 1, run step 2 of **Running Project** to start the simulation.
5. In the BlueSky GUI, select the **Nodes** tab on the lower-right side. Select a different simulation node to see the Austin Environment sim.

# Acknowledgements:

This project expands on the work of Marc Brittain
