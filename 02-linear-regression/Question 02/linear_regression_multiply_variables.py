import random
import matplotlib.pyplot as plt
import numpy as np
import csv

# Set learnig rate
LEARNING_RATE = 0.003
# Set number of digits do you wanto to consider in error convergence
ACCETABLE_ERROR = 0.1
# Set random thetha interval
THETA_INTERVAL = [-1, 1]
# Set max iterations
MAX_ITERATIONS = 1000


# x_vector = [1, x1 , x2], h_resp = theta1 + thetha2 * X1 + theta3 * x2
def hypothesis(thetas, x_vector):
    h_resp = 0
    for i in range(len(thetas)):
        h_resp += thetas[i] * x_vector[i]
    return h_resp


def average_error_function(data, thetas):
    m = len(data[0])
    average_error = 0
    error = [0] * (len(data))

    for i in range(m):
        x = []
        y = data[len(data) - 1][i]

        # x = [1, x1, x2]
        for j in range(len(data)):
            x.append(data[j - 1][i] if j else 1)

        # updated_value for thetas -> [(h(x) - y), ((h(x) - y) * x1), ((h(x) - y) * x2)]
        for j in range(len(data)):
            error[j] += ((hypothesis(thetas, x) - y) * x[j])
        average_error += ((hypothesis(thetas, x) - y) ** 2)

    for j in range(len(data)):
        error[j] = error[j] / m
    average_error /= 2 * m

    return average_error, error


def regression_learning(data):
    # thethas start with a random float value between the THETA_INTERVAL
    thetas = [0] * (len(data))
    thetas = [random.uniform(THETA_INTERVAL[0], THETA_INTERVAL[1]) for _ in thetas]

    iterations = average_error = 0
    errors_list = []

    while True:
        iterations += 1
        last_average_error = average_error
        average_error, error = average_error_function(data, thetas)
        # print("Average Error -->", average_error)
        errors_list.append(average_error)
        for i in range(len(thetas)):
            thetas[i] = thetas[i] - LEARNING_RATE * error[i]

        if MAX_ITERATIONS is not None:
            if iterations >= MAX_ITERATIONS:
                break
        else:
            if iterations > 2 and last_average_error - average_error <= ACCETABLE_ERROR:
                break
    print("To converge, the average error founding is:", average_error,
          "with the following parameters, learning rate:", LEARNING_RATE,
          "and accetable error is:", ACCETABLE_ERROR, ",intials thetas is randomic in interval:",
          THETA_INTERVAL, "the following number of iterations were needed:",
          iterations, "and final thetas is:", thetas)

    return errors_list, iterations


def plot_graph_coust_iterations(avgErrors, iterations):
    plt.title("Custo X Iteração : - Taxa de Aprendizado: " + str(LEARNING_RATE))
    plt.plot(list(range(iterations)), avgErrors)
    plt.xlabel('Iteração')
    plt.ylabel('Error')
    plt.show()


def main():
    data = [[], [], []]

    with open("./data2.txt") as csvfile:
        data_file = csv.reader(csvfile, delimiter=',', quotechar="|")
        for row in data_file:
            data[0].append(float(row[0]))
            data[1].append(float(row[1]))
            data[2].append(float(row[2]))

    # calculate feature mean
    area_mean = np.mean(data[0])
    room_mean = np.mean(data[1])

    # calculate feature standard deviation
    area_std = np.std(data[0])
    room_std = np.std(data[1])

    # Feature Normalization
    # Feature - feature_mean
    data[0] = [area - area_mean for area in data[0]]
    data[1] = [room - room_mean for room in data[1]]

    # Feature / feature standard deviation
    data[0] = [area / area_std for area in data[0]]
    data[1] = [room / room_std for room in data[1]]

    error_list, iterations = regression_learning(data)
    plot_graph_coust_iterations(error_list, iterations)


if __name__ == '__main__':
    main()
