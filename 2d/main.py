import argparse

import numpy as np
from phi_list import beta_phi_prime, beta_psi_prime, beta_psi_two_prime
from solver import NASA_beta

def ret_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta", default=1.2, help="This value should be greater than 1.",
    )
    parser.add_argument(
        "--num_iter", default=1000, help="The number of iteration of the algorithm.",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args   = ret_arg()

    x = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 500)
    # y = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 500)
    y_cont = np.vstack(
        [
            np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 490), 
            np.random.uniform(-50, 50, (10, 2))
        ], 
    )

    phi_prime_beta = beta_phi_prime(beta=args.beta)
    psi_prime_beta = beta_psi_prime(beta=args.beta)
    psi_two_prime_beta = beta_psi_two_prime(beta=args.beta)
    nasa_beta = NASA_beta(
        x_=x,
        y_=y_cont,
        Lambda=2,
        phi_prime=phi_prime_beta.value,
        psi_prime=psi_prime_beta.value,
        psi_two_prime=psi_two_prime_beta.value,
        z = 10,
    )
    PI, C = nasa_beta.search_optimal()
    print(PI[0]) # The last 10 elements have a value of 0, which means that nothing is being sent.