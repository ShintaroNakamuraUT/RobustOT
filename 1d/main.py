import argparse

import numpy as np
from phi_list import beta_phi_prime, beta_psi_prime, beta_psi_two_prime
from scipy import stats
from solver import NASA_beta

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta", default=1.5, help="This value should be greater than 1."
    )
    parser.add_argument(
        "--num_iter", default=1000, help="The number of iteration of the algorithm."
    )
    args = parser.parse_args()
    beta = args.beta
    num_iter = args.num_iter

    x = np.linspace(-10, 10, num=500)
    norm1 = stats.norm.pdf(x, loc=0, scale=1)

    y = np.linspace(-10, 10, num=500 - 5)
    norm2 = stats.norm.pdf(y, loc=0, scale=1)
    norm2 = np.concatenate([norm2, np.array([70, 70, 70, 70, 70])])
    print(norm1.size)
    print(norm2.size)

    phi_prime_beta = beta_phi_prime(beta=beta)
    psi_prime_beta = beta_psi_prime(beta=beta)
    psi_two_prime_beta = beta_psi_two_prime(beta=beta)

    nasa_beta = NASA_beta(
        x_=norm1,
        y_=norm2,
        Lambda=1,
        num_iter=num_iter,
        phi_prime=phi_prime_beta.value,
        psi_prime=psi_prime_beta.value,
        psi_two_prime=psi_two_prime_beta.value,
    )
    PI, C = nasa_beta.search_optimal()

    """You can see that the transportation cost is almost equal to zero. 
       This means that we have succeeded in ignoring the outliers."""
    print(sum(sum(PI * C)))
