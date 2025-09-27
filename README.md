# Puma Optimizer (PO)

## Overview

Puma Optimizer is a metaheuristic algorithm inspired by the intelligent hunting behavior and life of pumas. The algorithm's core strength lies in its unique and intelligent mechanism for switching between exploration (searching new areas) and exploitation (refining existing good solutions), allowing it to automatically adapt to the problem it's solving.

## How It Works

### 1. Initialization
**First**, the algorithm creates an initial random population of solutions. In the context of the algorithm, the entire search space is considered the puma's territory.

- Each member of the population, or solution (Xi), is considered a **female puma**
- The best solution found so far is designated as the **male puma** (Puma_male)

### 2. Puma Intelligence: The Phase Change Mechanism
This is the most innovative part of the PO algorithm. For the first three iterations, the puma is considered **"unexperienced"** and after that, it becomes **"experienced"**. This dictates how it chooses between exploration and exploitation.

#### Unexperienced Phase (First 3 Iterations)
In its early life, a puma is unfamiliar with its territory and lacks experience, so it performs exploration and exploitation simultaneously.

- **Simultaneous Operation**: For the first three iterations, both the exploration and exploitation phases are executed to gather initial "experience"
- **Scoring**: The performance of each phase is scored using two functions, f₁ and f₂. These functions measure the improvement in the cost (solution quality) over these initial runs
- **Final Score Calculation**: At the end of the third iteration, a final score for both exploration (Score_Explore) and exploitation (Score_Exploit) is calculated by combining their respective f₁ and f₂ scores
- **Population Update**: The solutions generated from both phases are combined, and only the best ones are kept to form the new population for the next iteration

#### Experienced Phase (Iterations 4 Onward)
After three generations, the puma has enough experience to intelligently choose only one phase per iteration. The choice is based on a scoring system that uses three functions:

- **Function f₁ (Intensification)**: This function rewards the phase that has recently shown a significant improvement in the best solution. It is calculated as f₁ᵗ = PF₁ · |Cost_old - Cost_new| / Tᵗ for each phase
- **Function f₂ (Resonance)**: This function rewards the phase that has shown consistent, sequential improvement over the last few times it was chosen
- **Function f₃ (Diversity)**: To avoid getting stuck, this function increases the score of a phase every time it is not selected. Once a phase is selected, its f₃ score resets to zero

Based on a weighted sum of these three scores, the algorithm decides whether to enter the exploration or exploitation stage. If Score_Exploit ≥ Score_Explore, it enters the exploitation phase; otherwise, it enters exploration.

### 3. Exploration Phase (Searching New Areas)
When the exploration phase is selected, pumas search their territory for new food sources. The population is first sorted based on cost. Then, for each puma, one of two strategies is chosen based on a random number:

- **Strategy 1 (Random Search)**: The puma jumps to a completely new random position within the search space boundaries. This is represented by the equation: Z_{i,G} = R_Dim × (Ub - Lb) + Lb
- **Strategy 2 (Differential Movement)**: The puma updates its position based on the locations of six other randomly selected pumas from the population

An adaptive mechanism is used to update the dimensions of the current solution. Initially, high-quality solutions undergo few changes, while lower-quality solutions are changed more significantly to explore the space better.

### 4. Exploitation Phase (Attacking Prey)
In the exploitation phase, pumas use two distinct hunting methods: sprinting and ambushing. The algorithm simulates this by randomly choosing between two main strategies to improve existing solutions:

- **Running Strategy**: This simulates a puma's fast run toward prey. The solution is updated based on the mean of all solutions in the population
- **Ambush Strategy**: This simulates a puma ambushing its prey and consists of two sub-strategies:
  - **Short Jumps**: The puma moves toward the hunts of other pumas (other good solutions)
  - **Long Jumps**: The puma leaps toward the best puma's hunt (the best solution found so far, Puma_male)

These updates involve several equations incorporating random numbers, exponential functions, and cosine functions to model the complex movements.

### 5. Update and Termination
After either the exploration or exploitation phase is complete, the newly generated solutions are evaluated. A new solution replaces the current one only if it has a better cost (is a better solution). The algorithm updates the position of the best overall solution (Puma_male) if a new best is found. This entire process repeats until a maximum number of iterations is reached, at which point the algorithm returns the final best solution.

## Using PO for Deep Learning Tasks

You can use this optimizer to train a neural network, although it works differently from gradient-based optimizers like Adam or SGD. This approach is a type of neuroevolution or "black-box" optimization.

### General Approach

**1. Define the Cost Function**: Your cost_function will be the function that evaluates your neural network. It should:
- Take a single vector `params` as input. This vector contains all the flattened weights and biases of your network
- Inside the function, reshape `params` and set them as the weights and biases of your neural network model
- Run your training or validation dataset through the network to calculate the loss (e.g., Mean Squared Error or Cross-Entropy)
- Return this single loss value. The optimizer will try to minimize it

**2. Set Up the Optimizer**:
- `dim`: This will be the total number of trainable parameters in your neural network. You'll need to calculate this by summing the sizes of all weight matrices and bias vectors
- `lower_bound` and `upper_bound`: These define the search range for each weight and bias. A common choice is a small range like [-1, 1]
- `cost_function`: Pass the function you created in step 1

**3. Run and Apply**:
- Run `po_optimizer.optimize()`
- The `best_solution` it returns will be the optimal set of weights and biases it found
- You then take this vector, reshape it, and set it as the final parameters for your trained model

**Important Consideration**: Metaheuristic algorithms like PO are generally much slower for training deep learning models than traditional gradient-based methods, especially on large datasets. However, they can be very powerful for problems where gradients are difficult or impossible to compute, or for exploring highly complex and non-convex loss landscapes.

## Code-to-Paper Mapping

Here is a breakdown of the main sections of the code, with references to where you can find the corresponding logic in the research paper.

### Class Initialization (PumaOptimizer)
- **Algorithm Parameters** (PF1, PF2, PF3, U, L, alpha): The default values for these parameters are taken directly from Table 3: "Parameter settings of optimization algorithms..."
- **State Variables** (pumas, best_puma, etc.): The concept of the population as "pumas" and the best solution as the "male puma" is introduced in Section 3.2: "Mathematical model"

### The _exploration() Method
This entire method implements the logic described in **Section 3.2.2: "Exploration"**.

- **Strategy Selection**: The `if np.random.rand() > 0.5:` check in the code corresponds to the choice between the two exploration strategies mentioned just before Equation (25)
  - The first strategy (random jump) is **Equation (25)**
  - The second strategy (differential movement) is the "Otherwise" case in **Equation (25)**, with the G parameter defined in Equation (26)
- **Adaptive Update**: The mechanism for creating the new solution X_new and adaptively updating its dimensions is described in Equation (27). The rule for increasing the U parameter when a solution is not improved comes from Equations (28-30)

### The _exploitation() Method
This method implements the logic from **Section 3.2.3: "Exploitation"**.

- **Strategy Selection**: The main if/else logic that chooses between the "Running Strategy" and the "Ambush Strategy" (which includes short and long jumps) is a direct implementation of Equation (32)
- **Helper Equations**: The calculations for the variables used in the exploitation strategies are found in these equations:
  - R is calculated using Equation (34)
  - F1 is calculated using Equation (35)
  - F2 is calculated using Equation (36), which itself uses w and v from Equations (37) and (38)

### The optimize() Method
This is the main driver that orchestrates the entire algorithm, and its logic comes from **Section 3.2.1: "Puma intelligence (phase change mechanism)"**.

- **Unexperienced Phase (first 3 iterations)**: This logic is detailed in Section 3.2.1.1: "Unexperienced phase". The scoring calculations performed at the end of iteration 3 are based on **Equations (1-12)**
- **Experienced Phase (iterations 4+)**: This logic follows the description in Section 3.2.1.2: "Experienced phase"
  - The decision `if self.score_exploit >= self.score_explore:` is the core of the intelligent phase selection
  - The updates to f1 and f3 scores are based on the principles of Equation (13/14) and Equation (17/18), respectively
  - The final update to score_explore and score_exploit at the end of the loop is a simplified implementation of the logic in Equations (19-24)

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{abdollahzadeh2024puma,
  title={Puma optimizer (PO): a novel metaheuristic optimization algorithm and its application in machine learning},
  author={Abdollahzadeh, Benyamin and Khodadadi, Nima and Barshandeh, Saeid and Trojovsk{\'y}, Pavel and Gharehchopogh, Farhad Soleimanian and El-kenawy, El-Sayed M and Abualigah, Laith and Mirjalili, Seyedali},
  journal={Cluster Computing},
  volume={27},
  pages={5235--5283},
  year={2024},
  publisher={Springer}
}
```

Paper DOI: [10.1007/s10586-023-04221-5](https://doi.org/10.1007/s10586-023-04221-5)