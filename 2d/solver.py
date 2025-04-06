import numpy as np

def dist(a, b):
    A = np.sum(a **2, )


class NASA_beta:
    def __init__(
        self,
        x_,
        y_,
        Lambda,
        phi_prime,
        psi_prime,
        psi_two_prime,
        z,
    ):
        self.x = x_
        self.y = y_
        self.Lambda = Lambda
        self.phi_prime = phi_prime
        self.psi_prime = psi_prime
        self.psi_two_prime = psi_two_prime
        self.beta = 1.2

        self.z = z

        self.num_iter = int(
            ((self.z/self.Lambda)*(self.beta - 1) - 1)/(
            (1/self.x.shape[0])**(self.beta - 1) + (1/self.y.shape[0])**(self.beta - 1)
            )
        )

    def search_optimal(self):
        print(self.x.shape[0])
        p = np.ones(self.x.shape[0]) / self.x.shape[0]
        q = np.ones(self.y.shape[0]) / self.y.shape[0]
        C = np.zeros((self.x.shape[0], self.y.shape[0]))
        for i in range(self.x.shape[0]):
            for j in range(self.y.shape[0]):
                C[i][j] = np.sqrt((self.x[i][0] - self.y[j][0])**2 + (self.x[i][1] - self.y[j][1])**2)
            

        theta_tilde = -C / self.Lambda

        theta_star = np.fmax(self.phi_prime(np.zeros_like(theta_tilde)), theta_tilde)
        plot_list = []

        prev = 0
        row_num = len(theta_star)
        column_num = len(theta_star.T)

        for i in range(self.num_iter):
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
            if i % 100 == 0:
                print(i)
                print(np.sum(self.psi_prime(theta_star) * C))
            

        pi_star = self.psi_prime(theta_star)
        return pi_star, C
