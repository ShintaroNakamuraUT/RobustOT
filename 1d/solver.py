import numpy as np


class NASA_beta:
    def __init__(
        self,
        x_,
        y_,
        Lambda,
        num_iter,
        phi_prime,
        psi_prime,
        psi_two_prime,
    ):
        self.x = x_
        self.y = y_
        self.Lambda = Lambda
        self.phi_prime = phi_prime
        self.psi_prime = psi_prime
        self.psi_two_prime = psi_two_prime
        self.beta = 1.5
        self.iter_num = num_iter

    def search_optimal(self):
        p = np.ones_like(self.x) / self.x.size
        q = np.ones_like(self.y) / self.y.size
        C = (
            np.repeat(self.y[None, :], self.x.size, axis=0)
            - np.repeat(self.x[None, :], self.y.size, axis=0).T
        ) ** 2

        theta_tilde = -C / self.Lambda

        theta_star = np.fmax(self.phi_prime(np.zeros_like(theta_tilde)), theta_tilde)
        plot_list = []

        prev = 0
        row_num = len(theta_star)
        column_num = len(theta_star.T)

        for i in range(8000):
            numerator = self.psi_prime(theta_star) @ np.ones_like(q) - p
            denominator = self.psi_two_prime(theta_star) @ np.ones_like(q).reshape(
                q.size, 1
            )

            tau = numerator / denominator.reshape(
                p.size,
            )

            dummy = np.array(
                [np.nanmax(theta_star[row]) for row in range(row_num)]
            ) - self.phi_prime(p)
            tau = np.fmax(tau, dummy)

            theta_tilde = theta_tilde - np.matmul(
                tau.reshape(tau.size, 1), np.ones_like(q).reshape(1, q.size)
            )

            theta_star = np.fmax(
                self.phi_prime(np.zeros_like(theta_tilde)), theta_tilde
            )

            numerator = (
                np.ones_like(p).reshape(1, p.size) @ self.psi_prime(theta_star) - q.T
            )
            denominator = np.ones_like(p).reshape(1, p.size) @ self.psi_two_prime(
                theta_star
            )

            sigma = numerator / denominator.reshape(
                q.size,
            )

            dummy = np.array(
                [np.nanmax(theta_star[:][column]) for column in range(column_num)]
            ) - self.phi_prime(q)
            sigma = np.fmax(sigma, dummy)

            theta_tilde = theta_tilde - np.matmul(
                np.ones_like(p).reshape(p.size, 1), sigma.reshape(1, sigma.size)
            )
            theta_star = np.fmax(
                self.phi_prime(np.zeros_like(theta_tilde)), theta_tilde
            )

            if i % 200 == 0:
                OT = sum(sum(self.psi_prime(theta_star) * C))
                if OT != 0 and prev != 0 and abs(OT - prev) / prev <= 1 / 1000:
                    break
                else:
                    prev = OT

        pi_star = self.psi_prime(theta_star)
        return pi_star, C
