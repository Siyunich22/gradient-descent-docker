import numpy as np

def gradient_descent(x, y, theta, learning_rate, iterations):
    """
    Выполняет градиентный спуск для минимизации ошибки по методу наименьших квадратов.

    :param x: numpy.ndarray, входные данные.
    :param y: numpy.ndarray, целевые значения.
    :param theta: numpy.ndarray, начальные коэффициенты модели.
    :param learning_rate: float, шаг обучения.
    :param iterations: int, количество итераций.
    :return: tuple (theta, cost_history), оптимизированные коэффициенты и история значений функции потерь.
    """
    m = len(y)  # Количество обучающих примеров
    cost_history = []  # История значений функции потерь

    for i in range(iterations):
        predictions = x.dot(theta)
        errors = predictions - y

        # Обновление коэффициентов
        gradients = (1/m) * x.T.dot(errors)
        theta -= learning_rate * gradients

        # Вычисление функции потерь
        cost = (1/(2*m)) * np.sum(errors ** 2)
        cost_history.append(cost)

        if i % 100 == 0:  # Печать прогресса каждые 100 итераций
            print(f"Iteration {i}: Cost {cost:.6f}")

    return theta, cost_history

if __name__ == "__main__":
    # Генерация данных
    np.random.seed(42)
    x = 2 * np.random.rand(100, 1)
    y = 4 + 3 * x + np.random.randn(100, 1)

    # Добавление столбца из единиц для bias (сдвиг)
    x_b = np.c_[np.ones((100, 1)), x]

    # Инициализация коэффициентов
    theta_initial = np.random.randn(2, 1)

    # Параметры
    learning_rate = 0.1
    iterations = 1000

    # Запуск градиентного спуска
    theta_optimized, cost_history = gradient_descent(x_b, y, theta_initial, learning_rate, iterations)

    print("Optimized theta:", theta_optimized)

    # Docker упаковка файла
    print("\nTo run in Docker:")
    print("1. Create a file named 'Dockerfile' in the project folder.")
    print("2. Add the following content:")
    print("\nFROM python:3.9\nCOPY . /app\nWORKDIR /app\nRUN pip install numpy\nCMD [\"python\", \"gradient_descent.py\"]")
    print("\n3. Build the Docker image:")
    print("docker build -t gradient-descent .")
    print("\n4. Run the Docker container:")
    print("docker run gradient-descent")
