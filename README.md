# Learning Assisted Hybrid Simulation-Optimization Model

The **Learning Assisted Hybrid Simulation-Optimization Model** provides a modular framework, applying a Reinforcement Learning approach to synchromodal transport in response to disruptions. The model can be used to train a learning agent and implement the trained agent to be compared with different policies or different training strategies.

This documentation includes a ready-to-use learning agent that has been trained for 50,000 episodes with a disruption profile located in the dataset folder.

## Installation

1. [Clone this repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository), or [download the zip file of the code](https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives#downloading-source-code-archives-from-the-repository-view) and unzip it.
2. Install [Gurobi](https://www.gurobi.com/).
3. Install [uv](https://docs.astral.sh/uv/).
4. Open a terminal window in the directory you cloned/unzipped into.
5. Run `uv sync`.
6. Run `uv run ui`.
7. A browser window should open automatically, otherwise copy the printed link.

## Run the Model

### 1. Model input
- Once the repository is cloned, there is a folder called "Datasets" consisting of default datasets required for running the model.
- The default dataset is provided automatically once the model is started. Users can also create their input dataset by following the format given in the default data.

### 2. Providing K-Best solutions
- In the "Datasets" folder, the K-best solution for the default demand is already given.
- If the user wants to generate a new demand file, the user needs to tick the box "Provide K-Best Solution" to generate the K-Best solution. The model will create a new file consisting of the demand with K-best solutions after "Next Step" button is clicked

### 3. Learning Agent Settings
- For training an agent, this settings is used to change hyperparameters of the agent
- For implementation, users can upload different Q-table (a product of trained agent) and choose different polices. The greedy policy utilizes the RL agent while the others are benchmark policies

### 4. Training an Agent
- Users can train a model from scratch and will get training output (Q-table, record of total cost, and record of total reward) in a default name.
- Since the training can take days to complete, users can stop the training and continue to train the agent by ticking the "Continue from previous training" box
- After ticking the "Continue from previous training" box, the user will have an option to upload the last trained agent (Q-table, along with the recorded total costs and reward).
- If users leave those options in the default files, the model will overwrite the Q-table with the default name. **Therefore it is important to rename the Q-table file (along with the recorded total costs and reward) if the user wants to save it.**

### 5. Executing Model Implementation
- After clicking the "Next Step" button in Simulation Settings tab, the model is ready to run. The user simply selects the tab "Execute Simulation" and the model will run until the desired number of simulations are completed
- The bar chart provide the total costs of each episode.

### 6. Comparing results
- In the "Results Comparison" tab, users can compare output from Model Implementation after running different policies or learning agent.
- The output of the Model Implementation are stored in csv_ouptut folder. Users can upload to files for comparison in the given box and determine the labell
- By switching to the "Policy Comparison" tab, users will get 5 charts of comparison of different cost parameters. The bar chart represent cost different between 2 polices.



---

## File Descriptions

### 1. `Config.py`
This file contains the configuration settings for the entire model, including simulation settings, training parameters, and file paths. Simulation settings include the duration, number of episodes, random seed, and other customizable options. Boolean switches are used to determine whether the simulation is for training or implementation purposes.

### 2. `Model_input.py`
This file reads input files and preprocesses data before passing it to the simulation module. Key data components include:

- **Network Data**: Terminal locations and distances, service line characteristics (schedule, route, travel speed, capacity).
- **Demand Data**: Shipments defined by origin, destination, time requirements, and volume.
- **Cost Parameters**: Storage, handling, travel, and delay costs.
- **Disruption Profiles**: Potential locations, durations, capacity reductions, and frequencies.

Other parameters, such as time windows and learning agent hyperparameters (epsilon, alpha, gamma), are also configured here.

### 3. `Helper_functions.py`
Utility functions for input data preprocessing and helper functions for managing simulations (e.g., simulation clock).

### 4. `Global_variables.py`
Defines global variables for simulation, which are accessed and updated by functions. It includes a reset function for episodic simulations.

### 5. `Policy_function.py`
Implements the policy for the Reinforcement Learning (RL) agent, which interacts with the Q-table. Available policies include:
- ε-greedy
- Greedy
- Always wait
- Always reassign

The ε-greedy policy balances exploration and exploitation during training.

### 6. `Extract_output.py`
Extracts shipment logs after each simulation, tracking observations such as shipment costs, actions taken, and volume transported.

### 7. `Optimization_module.py`
Solves a matching optimization problem to generate shipment plans. It defines the objective and constraints, and is executed through the `run_optimization` function. This requires the Gurobi package.

### 8. `Temp_plot.py`
Generates line charts showing rolling average costs and rewards during training to monitor progress. The `smoothing` variable in `config.py` determines the horizon for the rolling average.

### 9. `Simulation_module.py`
Contains classes and functions for discrete event simulation using Simpy. Key components include:

- **Mode (Class)**: Represents a service line, handling the route from port to inland terminals. Interacts with shipments during loading and unloading.
- **Shipment (Class)**: Represents transportation demand and is assigned to service lines for execution.
- **ServiceDisruption (Class)**: Generates disruptions based on profiles, halting operations at disrupted locations.
- **DemandDisruption (Class)**: Affects shipments at origin terminals without disrupting the broader network.
- **MatchingModule (Class)**: Manages offline and online planning, generating shipment plans and updating service paths during disruptions.
- **ReinforcementLearning (Class)**: Selects actions using the Q-table, calculates costs as rewards, and updates the Q-table accordingly.

### 10. `Service_to_path.ipynb`
Converts a network of terminals and service lines into possible paths for the optimization algorithm. It calculates the associated costs for each path.

### 11. `K-Best.ipynb`
Stores potential solutions for each shipment in a solution pool to minimize the frequency of triggering the optimization module. Uses a rolling horizon approach to update service capacities and provide solutions for varying announcement times.

### 12. `Main_file.ipynb`
Runs the simulation and records observations. The simulation can be used for either training or implementation, as specified in `config.py`. Key steps include:

- Setting the policy to ε-greedy for training.
- Running thousands of episodes and updating the Q-table after each.
- Pausing and resuming training by setting `start_from_0` to `False`.
- For implementation, reducing `number_of_simulation` for quicker testing and comparing results.

## Datasets

All required datasets are located in the `Datasets` folder. The `Data Structure` file explains each dataset. Additional datasets include:
- **Possible Paths**: Generated by running `Service_to_path.ipynb`.
- **Shipment Request with Solution Pool**: Generated by executing `K-Best.ipynb`.
- **Disruption Sets**: Vary by occurrence probability.

The trained Q-table can be found in the `q_table` folder (`q_table_200_50000_eps.pkl`).

---

## Steps to Execute the Model

1. **Prepare Required Datasets**:
    - Network data, including general and mode-specific networks.
    - Service schedules with properties (speed, capacity, etc.).
    - Demand data.
    - Disruption profiles.
    - Cost parameters.

2. **Set Up Configuration**:
    Configure simulation settings, file paths, and whether to run in training or implementation mode in `Config.py`.

3. **Generate Paths**:
    If using a new network file, run `Service_to_path.ipynb` to generate possible paths.

4. **Generate Solution Pool**:
    If using the solution pool approach, run `K-Best.ipynb` to store possible solutions for each shipment.

5. **Run the Simulation**:
    Execute `Main_file.ipynb` to run episodic simulations.

6. **Resume Training**:
    To resume training from a previous run, set `start_from_0` in `Config.py` to `False`.

7. **Monitor Training Progress**:
    Use `Temp_plot.py` to view the rolling average costs and rewards during training.

8. **Extract Observations**:
    For implementation, observations from each simulation episode are saved in the `csv_output` folder. Shipment logs are stored in the `shipment_logs` folder if `extract_shipment_output` is set to `True` in `Config.py`.

---

For more details on how files interact, refer to the Flow Diagram file in the project.
