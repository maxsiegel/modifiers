import numpy as np
import argparse

def func_to_iterate_over(args):
    # idx, shared_array = args
    # np.array(shared_array, dtype=np.float64)
    results = []
    # print(shared_array)
    # for params in shared_array[idx]:
    for params in args:
        thetas = params[:5]
        log_l_arr = np.zeros((len(possible_phi), len(possible_alpha)))

        results = []
        for phi in possible_phi:
            for alpha in possible_alpha:
                for pred in predicates:
                    P_grid = np.zeros((len(states), len(intensifiers)))
                    for i in range(len(states)):
                        s = states[i]
                        for j in range(len(intensifiers)):
                            P_grid[i][j] = (1 - phi) * theta_U_inf[thetas[j]][int(
                                (s - S) * 10)]
                    P_grid = P_grid + phi * np.array([
                        U_soc[(intensifiers[j], pred)]
                        for j in range(len(intensifiers))
                    ])
                    P_grid = alpha * P_grid
                    P_grid = softmax_rowwise(P_grid)
                    # multiply row i of P_grid by P_state[states[i]] u
                    # swap rows and columns of P_grid
                    # so each row is the same intensifier
                    P_grid = np.array(P_grid).T
                    P_grid = P_grid * P_state
                    # P_grid is P_s1(w|s,phi,alpha)*P(s) unnormalized
                    for i in range(len(intensifiers)):
                        w = intensifiers[i]
                        P_l1[(w, pred)] = P_grid[i]  # row with lots of states
                        # normalize over states adds up to 1
                        P_l1[(w, pred)] = P_l1[(w, pred)] / np.sum(P_l1[(w, pred)])
                log_likelihood = 0
                for w in intensifiers:
                    for pred in predicates:
                        log_likelihood += np.sum(np.log(P_l1[(w, pred)][int((s - S) * 10)]) for s in measured_values[(w, pred)])
                log_l_arr[int((phi - Phi) * 10)][int(
                    (alpha - Alpha) * 10 / 3)] = log_likelihood
                results.append((log_l_arr, thetas))
    # return (log_l_arr, thetas)
    # with open('results/' + str(hash(tuple(args))), 'wb') as f:
    #     pickle.dump(results, f)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.tuple_file, "r") as f:
        data = json.load(f)

    results = func_to_iterate_over(args)
    print(results)

    return results
