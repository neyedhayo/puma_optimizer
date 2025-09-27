"""
    Puma Optimizer (PO) implementation based on the paper
"""

import numpy as np
import time


class PumaOptimizer:
    def __init__(self, cost_function, dim, n_pop, max_iter, lower_bound, upper_bound, 
                 PF1=0.5, PF2=0.5, PF3=0.3, U=0.2, L=0.9, alpha=2.0):
        
        self.cost_function = cost_function
        self.dim = dim
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.lower_bound = np.full(dim, lower_bound)
        self.upper_bound = np.full(dim, upper_bound)
        
        # algorithm parameters
        self.PF1 = PF1
        self.PF2 = PF2
        self.PF3 = PF3
        self.U = U
        self.L = L
        self.alpha = alpha

        # initialize state variables
        self.pumas = np.zeros((self.n_pop, self.dim))
        self.puma_costs = np.full(self.n_pop, np.inf)
        self.best_puma = None
        self.best_puma_cost = np.inf
        
        # phase change mechanism variables
        self.score_explore = 1.0
        self.score_exploit = 1.0
        self.f1_explore, self.f1_exploit = 0.0, 0.0
        self.f3_explore, self.f3_exploit = 0.0, 0.0
        self.alpha_explore, self.alpha_exploit = 0.5, 0.5
        self.T_explore, self.T_exploit = 1.0, 1.0 
        self.seq_cost_explore = [0, 0, 0]
        self.seq_cost_exploit = [0, 0, 0]

    def _initialize_population(self):
        """Initializes the puma population randomly within the search bounds."""
        
        for i in range(self.n_pop):
            self.pumas[i] = self.lower_bound + np.random.rand(self.dim) * (self.upper_bound - self.lower_bound)
            self.puma_costs[i] = self.cost_function(self.pumas[i])
        
        best_idx = np.argmin(self.puma_costs)
        self.best_puma = self.pumas[best_idx].copy()
        self.best_puma_cost = self.puma_costs[best_idx]

    def _exploration(self):
        """Performs the exploration phase (Sec 3.2.2 in the paper)."""
        
        sorted_indices = np.argsort(self.puma_costs)
        pumas_sorted = self.pumas[sorted_indices]
        costs_sorted = self.puma_costs[sorted_indices]
        
        new_pumas = pumas_sorted.copy()
        new_costs = costs_sorted.copy()
        
        U_current = self.U
        p = (1 - U_current) / self.n_pop # Eq. (29)

        for i in range(self.n_pop):
            a, b, c, d, e, f = np.random.choice(self.n_pop, 6, replace=False)
            
            Z = np.zeros(self.dim)
            if np.random.rand() > 0.5:
                Z = self.lower_bound + np.random.rand(self.dim) * (self.upper_bound - self.lower_bound) # Eq. (25)
            else:
                G = 2 * np.random.rand() - 1 # Eq. (26)
                Z = (self.pumas[a] + G * (self.pumas[a] - self.pumas[b]) + 
                     G * ((self.pumas[a] - self.pumas[b]) - (self.pumas[c] - self.pumas[d])) +
                     ((self.pumas[c] - self.pumas[d]) - (self.pumas[e] - self.pumas[f])))
            
            Z = np.clip(Z, self.lower_bound, self.upper_bound)
            
            # update solution dimensions based on U_current
            X_new = new_pumas[i].copy()
            j_rand = np.random.randint(0, self.dim)
            for j in range(self.dim):
                if j == j_rand or np.random.rand() <= U_current:
                    X_new[j] = Z[j]
            
            cost_new = self.cost_function(X_new)
            if cost_new < new_costs[i]:
                new_pumas[i] = X_new
                new_costs[i] = cost_new
            else:
                U_current += p # Eq. (30)
        
        return new_pumas, new_costs

    def _exploitation(self):
        """Performs the exploitation phase (Sec 3.2.3 in the paper)."""
        new_pumas = self.pumas.copy()
        new_costs = self.puma_costs.copy()

        for i in range(self.n_pop):
            # calculate R, F1, F2 from Eqs. (34-38)
            R = 2 * np.random.rand() - 1 
            F1 = np.random.randn(self.dim) * np.exp(2 - self.current_iter * (2 / self.max_iter))
            w = np.random.randn(self.dim)
            v = np.random.randn(self.dim)
            F2 = w * (v**2) * np.cos((2 * np.random.rand()) * w)
            
            X_new = np.zeros(self.dim)
            
            # strategy Selection from Eq. (32) ----- TRICKY?
            if np.random.rand() >= 0.5: # running Strategy
                mean_sol = np.mean(self.pumas, axis=0)
                X_new = self.pumas[i] - self.alpha * (mean_sol / self.n_pop) + (self.best_puma - self.pumas[i])
            else: # ambush Strategy
                if np.random.rand() >= self.L: # short jumps
                    r1_idx = np.random.randint(0, self.n_pop)
                    s = np.random.choice([0, 1])
                    X_new = self.pumas[i] + ((-1)**s) * F1 * (self.pumas[r1_idx] - self.pumas[i])
                else: # long jumps
                    r2_idx = np.random.randint(0, self.n_pop)
                    X_new = (F1 * self.pumas[i] + F2 * (1 - R) * self.best_puma)
            
            X_new = np.clip(X_new, self.lower_bound, self.upper_bound)
            
            cost_new = self.cost_function(X_new)
            if cost_new < new_costs[i]:
                new_pumas[i] = X_new
                new_costs[i] = cost_new
        
        return new_pumas, new_costs

    def optimize(self):
        """Main optimization loop for the Puma Optimizer."""
        self._initialize_population()
        initial_best_cost = self.best_puma_cost

        for iter_num in range(1, self.max_iter + 1):
            self.current_iter = iter_num
            
            # --- unexperienced Phase (First 3 iterations) (Sec 3.2.1.1) ---
            if iter_num <= 3:
                explore_pumas, explore_costs = self._exploration()
                exploit_pumas, exploit_costs = self._exploitation()
                
                best_explore_cost = np.min(explore_costs)
                best_exploit_cost = np.min(exploit_costs)
                
                if iter_num == 1:
                    self.seq_cost_explore[0] = abs(initial_best_cost - best_explore_cost)
                    self.seq_cost_exploit[0] = abs(initial_best_cost - best_exploit_cost)
                else:
                    self.seq_cost_explore[iter_num-1] = abs(best_explore_cost - self.prev_best_explore)
                    self.seq_cost_exploit[iter_num-1] = abs(best_exploit_cost - self.prev_best_exploit)

                self.prev_best_explore = best_explore_cost
                self.prev_best_exploit = best_exploit_cost

                # Combine results and select the best
                all_pumas = np.vstack((explore_pumas, exploit_pumas))
                all_costs = np.hstack((explore_costs, exploit_costs))
                best_indices = np.argsort(all_costs)[:self.n_pop]
                self.pumas = all_pumas[best_indices]
                self.puma_costs = all_costs[best_indices]

                # update global best
                if self.puma_costs[0] < self.best_puma_cost:
                    self.best_puma_cost = self.puma_costs[0]
                    self.best_puma = self.pumas[0].copy()

                # at the end of iteration 3, calculate initial scores
                if iter_num == 3:
                    f1_explore = self.PF1 * (self.seq_cost_explore[0] / 1.0)
                    f1_exploit = self.PF1 * (self.seq_cost_exploit[0] / 1.0)
                    f2_explore = self.PF2 * (np.sum(self.seq_cost_explore) / 3.0)
                    f2_exploit = self.PF2 * (np.sum(self.seq_cost_exploit) / 3.0)
                    self.score_explore = (self.PF1 * f1_explore) + (self.PF2 * f2_explore)
                    self.score_exploit = (self.PF1 * f1_exploit) + (self.PF2 * f2_exploit)
            
            # --- experienced phase (After 3 iterations) (Sec 3.2.1.2) ---
            else:
                cost_before_phase = self.best_puma_cost
                
                if self.score_exploit >= self.score_explore:
                    chosen_phase = 'exploit'
                    self.pumas, self.puma_costs = self._exploitation()
                else:
                    chosen_phase = 'explore'
                    self.pumas, self.puma_costs = self._exploration()
                
                current_best_idx = np.argmin(self.puma_costs)
                if self.puma_costs[current_best_idx] < self.best_puma_cost:
                    self.best_puma_cost = self.puma_costs[current_best_idx]
                    self.best_puma = self.pumas[current_best_idx].copy()

                cost_after_phase = self.best_puma_cost
                
                # update scores based on performance (Eqs. 13-24)
                if chosen_phase == 'exploit':
                    self.f1_exploit = self.PF1 * abs((cost_before_phase - cost_after_phase) / self.T_exploit)
                    self.f3_exploit = 0
                    self.f3_explore += self.PF3
                    self.T_exploit = 1
                    self.T_explore += 1
                else:
                    self.f1_explore = self.PF1 * abs((cost_before_phase - cost_after_phase) / self.T_explore)
                    self.f3_explore = 0
                    self.f3_exploit += self.PF3
                    self.T_explore = 1
                    self.T_exploit += 1
                
                # update alpha weights for scoring
                if self.score_exploit > self.score_explore:
                    self.alpha_exploit = 0.99
                    self.alpha_explore = max(0.01, self.alpha_explore - 0.01)
                else:
                    self.alpha_explore = 0.99
                    self.alpha_exploit = max(0.01, self.alpha_exploit - 0.01)

                # update final scores for next iteration's decision
                self.score_explore = (self.alpha_explore * self.f1_explore) + ((1 - self.alpha_explore) * self.f3_explore)
                self.score_exploit = (self.alpha_exploit * self.f1_exploit) + ((1 - self.alpha_exploit) * self.f3_exploit)

            print(f"Iteration {iter_num}/{self.max_iter}, Best Cost: {self.best_puma_cost:.6f}")

        return self.best_puma, self.best_puma_cost

if __name__ == '__main__':
    def sphere_function(x):
        return np.sum(x**2)

    DIMS = 10
    POP_SIZE = 30
    MAX_ITERATIONS = 100
    LOWER_BOUND = -100
    UPPER_BOUND = 100

    po_optimizer = PumaOptimizer(
        cost_function=sphere_function,
        dim=DIMS,
        n_pop=POP_SIZE,
        max_iter=MAX_ITERATIONS,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND
    )

    start_time = time.time()
    best_solution, best_cost = po_optimizer.optimize()
    end_time = time.time()

    print("\n--- Optimization Finished ---")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Best Solution: {best_solution}")
    print(f"Best Cost: {best_cost}")