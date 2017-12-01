from model import Knn


def main():
    # Quantity of generated points
    n = 1000
    # Accuracy measurements for given k
    attempts = 5
    # Step in accuracy measurements
    step = 10
    # Coefficient of disorder of points in data sets
    disorder_coeff = 0.04

    # Accuracy of all data sets
    accuracy_data = []
    labels = ['podział liniowy', 'szachownica 2x2', 'szachownica 3x3', 'szachownica 4x4', 'szachownica 5x5']

    # Data set with linear points division
    lin_data = Knn()
    lin_data.points_gen_lin(n, disorder_coeff)
    lin_data.plot_data_set()
    accuracy_data.append(lin_data.knn_accuracy_list(step, attempts=attempts))
    Knn.plot_decision_boundaries(lin_data.x, lin_data.y, 430)

    # Data set chessboard 2x2
    squares_4 = Knn()
    squares_4.points_gen_chessboard(n, 2, disorder_coeff)
    squares_4.plot_data_set()
    accuracy_data.append(squares_4.knn_accuracy_list(step, attempts=attempts))
    Knn.plot_decision_boundaries(squares_4.x, squares_4.y, 210)

    # Data set chessboard 3x3
    squares_9 = Knn()
    squares_9.points_gen_chessboard(n, 3, disorder_coeff)
    squares_9.plot_data_set()
    accuracy_data.append(squares_9.knn_accuracy_list(step, attempts=attempts))
    Knn.plot_decision_boundaries(squares_9.x, squares_9.y, 70)

    # Data set chessboard 4x4
    squares_16 = Knn()
    squares_16.points_gen_chessboard(n, 4, disorder_coeff)
    squares_16.plot_data_set()
    accuracy_data.append(squares_16.knn_accuracy_list(step, attempts=attempts))
    Knn.plot_decision_boundaries(squares_16.x, squares_16.y, 50)

    # Data set chessboard 5x5
    squares_25 = Knn()
    squares_25.points_gen_chessboard(n, 5, disorder_coeff)
    squares_25.plot_data_set()
    accuracy_data.append(squares_25.knn_accuracy_list(step, attempts=attempts))
    Knn.plot_decision_boundaries(squares_25.x, squares_25.y, 10)

    # Accuracy of all data sets plot
    Knn.plot_knn_accuracy(accuracy_data, labels, 5)

if __name__=='__main__':
    main()