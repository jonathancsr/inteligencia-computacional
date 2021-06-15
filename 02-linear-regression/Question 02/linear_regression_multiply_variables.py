import random
import matplotlib.pyplot as plt
import numpy as np
import csv


# # Set learnig rate
# LEARNING_RATE = 0.01
# # Set number of digits do you wanto to consider in error convergence
# ERROR_SIGNIFICANT_DIGITS = 6
# # Set random thetha interval
# THETA_INTERVAL = [-1, 1]
#
#
# # x_vector = [1, x], h_resp = theta1 + thetha2 * X
# def hypothesis(thetas, x_vector):
#     h_resp = 0
#     for i in range(len(thetas)):
#         h_resp += thetas[i] * x_vector[i]
#     return h_resp
#
#
# def average_error_function(data, thetas):
#     error = [0] * (len(data))
#     average_error = 0
#     m = len(data[0])
#     for i in range(m):
#         x = []
#         y = data[len(data) - 1][i]
#
#         # x = [1, x]
#         for j in range(len(data)):
#             x.append(data[j - 1][i] if j else 1)
#
#         # updated_value for thetas -> [(h(x) - y), ((h(x) - y) * x)]
#         for j in range(len(data)):
#             error[j] += ((hypothesis(thetas, x) - y) * x[j])
#         average_error += ((hypothesis(thetas, x) - y) ** 2)
#
#     for j in range(len(data)):
#         error[j] = error[j] / m
#
#     average_error /= 2 * m
#
#     return average_error, error
#
#
# def regression_learning(data):
#     # thethas start with a random float value between the THETA_INTERVAL
#     thetas = [random.uniform(THETA_INTERVAL[0], THETA_INTERVAL[1])] * (len(data))
#     iterations = average_error = 0
#     error_list = []
#
#     while True:
#         iterations += 1
#         last_average_error = average_error
#         average_error, error = average_error_function(data, thetas)
#         error_list.append(average_error)
#         for i in range(len(thetas)):
#             thetas[i] = thetas[i] - LEARNING_RATE * error[i]
#
#         if round(average_error, ERROR_SIGNIFICANT_DIGITS) == round(last_average_error, ERROR_SIGNIFICANT_DIGITS):
#             break
#
#     gradient = [
#         [min(data[0]), max(data[0])]
#         , [hypothesis(thetas, [1, min(data[0])]), hypothesis(thetas, [1, max(data[0])])]
#     ]
#
#     print("To converge, the average error founding is:", round(average_error, ERROR_SIGNIFICANT_DIGITS),
#           "with the following parameters, learning rate:", LEARNING_RATE,
#           "and significant digits:", ERROR_SIGNIFICANT_DIGITS, ",intials thetas is randomic in interval:",
#           THETA_INTERVAL, "the following number of iterations were needed:",
#           iterations)
#
#     return gradient, error_list, iterations
#
#
# def plot_linear_graph(data, gradient):
#     plt.title("Population X Profit")
#     plt.plot(data[0], data[1], 'rx')
#     plt.plot(gradient[0], gradient[1])
#     plt.xlabel('População por 10.000')
#     plt.ylabel('Lucro por 10.000')
#     plt.show()
#
#
# def plot_graph_coust_iterations(avgErrors, iterations):
#     plt.title("Custo X Iteração : ")
#     plt.plot(list(range(iterations)), avgErrors)
#     plt.xlabel('Iteração')
#     plt.ylabel('Error')
#     plt.show()
#

def main():
    data = [[], [], []]

    with open("./data2.txt") as csvfile:
        data_file = csv.reader(csvfile, delimiter=',', quotechar="|")
        for row in data_file:
            data[0].append(float(row[0]))
            data[1].append(float(row[1]))
            data[2].append(float(row[2]))

    area_mean = np.mean(data[0])
    room_mean = np.mean(data[1])

    area_std = np.std(data[0])
    room_std = np.std(data[1])

    data[0] = [area - area_mean for area in data[0]]
    data[1] = [room - room_mean for room in data[1]]

    data[0] = [area - area_std for area in data[0]]
    data[1] = [room - room_std for room in data[1]]

    # regression, error_list, iterations = regression_learning(data)
    # plot_linear_graph(data, regression)
    # plot_graph_coust_iterations(error_list, iterations)


if __name__ == '__main__':
    main()
