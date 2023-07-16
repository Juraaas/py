import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class KnnClass:
    def __init__(self, data, group_test=0):
        self.data = data
        self.data['group'] = ''  # przypisywanie grup do gatunkow irysow
        self.data.loc[:49, 'group'] = 'A'
        self.data.loc[50:99, 'group'] = 'B'
        self.data.loc[100:149, 'group'] = 'C'
        train, test = train_test_split(self.data, test_size=0.3)  # podzial na grupe testowa i treningowa
        self.test_copy = test.copy()  # robienie kopii
        self.data = train.copy()
        train.drop('group', axis=1, inplace=True)  # usuniecie kolumny z treningu
        self.x = train.values  # przeksztalcenie na macierz wartosci
        self.k = 3  # liczba sasiadow do uwzglednienia
        self.accuracy = 0
        test.drop('group', axis=1, inplace=True)  # usuniecie kolumny z testu
        if group_test == 0:
            self.group_test = test.values.tolist()  # przypisanie danych testowych do group_test
        else:
            self.group_test = group_test
        self.test_data = pd.DataFrame(self.group_test, columns=iris.feature_names)
        self.test_data['group'] = ''

    def euclides_distance_func(self):
        distance_list = []
        for test_sample in self.group_test:
            distances = np.linalg.norm(self.x - test_sample, axis=1)  # normalizacja wartości
            distance_list.append(distances)
        assert len(distance_list) == len(self.group_test)
        return distance_list

    def cosine_distance_func(self):
        distance_list = []
        for test_sample in self.group_test:
            dot_products = np.dot(self.x, test_sample)  # iloczyn skalarny
            norm_vector1 = np.linalg.norm(self.x, axis=1)  # normalizacja wektorów
            norm_vector2 = np.linalg.norm(test_sample)
            cosine_distances = 1 - dot_products / (norm_vector1 * norm_vector2)
            distance_list.append(cosine_distances)
        assert len(distance_list) == len(self.group_test)
        return distance_list

    def manhattan_distance_func(self):
        distance_list = []
        for test_sample in self.group_test:
            abs_diff = np.abs(self.x - test_sample)  # wart bezwzgledna z kazdego przykladu testowego
            manhattan_distances = np.round(np.sum(abs_diff, axis=1), 3)  # suma dla kazdego wiersza
            distance_list.append(manhattan_distances)
        assert len(distance_list) == len(self.group_test)
        return distance_list

    def knn_calculation(self, dist_func):
        copy_test_groups = self.test_copy['group'].values
        correct = 0
        dist = dist_func
        k = self.k
        for i in range(len(self.group_test)):
            self.data['distance_from_test'] = dist[i]  # przypisanie wynikow funkcji obliczajacej dystans do nowej kolumny
            data_temp = self.data.sort_values('distance_from_test', ascending=True)  # sortowanie wynikow
            k_nearest = data_temp[:k]  # wyodrebnienie k najblizszych punktow
            distances = k_nearest['distance_from_test']
            inverse_distances = np.zeros_like(distances)  # stworzenie tablicy na odwrotnosci
            non_zero_indices = distances > 0
            inverse_distances[non_zero_indices] = 1 / distances[non_zero_indices]  # obliczanie dla wartosci wiekszych od 0
            groups = k_nearest['group']
            groups = groups.values.tolist()  # wyodrebnienie k najblizszych grup do listy
            inverse_sum = np.sum(inverse_distances)
            weights = np.zeros_like(inverse_distances)
            if inverse_sum > 0:
                weights[non_zero_indices] = inverse_distances[non_zero_indices] / inverse_sum  # obliczanie wagi dla > 0
            group_dict = {
                'A': 0,
                'B': 0,
                'C': 0
            }
            for j in range(len(inverse_distances)):
                group_dict[groups[j]] += weights[j]  # przydzielenie wagi
            defined_group = max(group_dict, key=group_dict.get)  # przydzielenie do grupy z najwieksza waga
            self.test_data.loc[i, 'group'] = defined_group  # przypisania grupy do danej kolumny
            if defined_group == copy_test_groups[i]:  # sprawdzenie poprawnosci przypisania grupy
                correct += 1
        self.accuracy = np.round(correct / len(copy_test_groups) * 100, 2)  # zliczanie skutecznosci
        return None

    def plot_knn(self, dist_func):
        self.knn_calculation(dist_func)
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        a_group_data = self.data[self.data['group'] == 'A']
        b_group_data = self.data[self.data['group'] == 'B']
        c_group_data = self.data[self.data['group'] == 'C']
        color_dict = {
            'A': 'red',
            'B': 'blue',
            'C': 'black'
        }

        axs[0, 0].scatter(a_group_data['sepal length (cm)'], a_group_data['sepal width (cm)'], color='red')
        axs[0, 0].scatter(b_group_data['sepal length (cm)'], b_group_data['sepal width (cm)'], color='blue')
        axs[0, 0].scatter(c_group_data['sepal length (cm)'], c_group_data['sepal width (cm)'], color='black')
        axs[0, 0].scatter(self.test_data['sepal length (cm)'], self.test_data['sepal width (cm)'], color='gold')
        axs[0, 0].set_xlabel('length')
        axs[0, 0].set_ylabel('width')
        axs[0, 0].set_title('Sepal bez identyfikacji')

        axs[0, 1].scatter(a_group_data['petal length (cm)'], a_group_data['petal width (cm)'], color='red')
        axs[0, 1].scatter(b_group_data['petal length (cm)'], b_group_data['petal width (cm)'], color='blue')
        axs[0, 1].scatter(c_group_data['petal length (cm)'], c_group_data['petal width (cm)'], color='black')
        axs[0, 1].scatter(self.test_data['petal length (cm)'], self.test_data['petal width (cm)'], color='gold')
        axs[0, 1].set_xlabel('length')
        axs[0, 1].set_ylabel('width')
        axs[0, 1].set_title('Petal bez indentyfikacji')

        axs[1, 0].scatter(a_group_data['sepal length (cm)'], a_group_data['sepal width (cm)'], color='red')
        axs[1, 0].scatter(b_group_data['sepal length (cm)'], b_group_data['sepal width (cm)'], color='blue')
        axs[1, 0].scatter(c_group_data['sepal length (cm)'], c_group_data['sepal width (cm)'], color='black')
        for value in range(len(self.test_data)):
            axs[1, 0].scatter(self.test_data['sepal length (cm)'][value], self.test_data['sepal width (cm)'][value],
                              color=color_dict[self.test_data['group'][value]])
        axs[1, 0].set_xlabel('length')
        axs[1, 0].set_ylabel('width')
        axs[1, 0].set_title('Sepal po indentyfikacji')

        axs[1, 1].scatter(a_group_data['petal length (cm)'], a_group_data['petal width (cm)'], color='red')
        axs[1, 1].scatter(b_group_data['petal length (cm)'], b_group_data['petal width (cm)'], color='blue')
        axs[1, 1].scatter(c_group_data['petal length (cm)'], c_group_data['petal width (cm)'], color='black')
        for value in range(len(self.test_data)):
            axs[1, 1].scatter(self.test_data['petal length (cm)'][value], self.test_data['petal width (cm)'][value],
                              color=color_dict[self.test_data['group'][value]])
        axs[1, 1].set_xlabel('length')
        axs[1, 1].set_ylabel('width')
        axs[1, 1].set_title('Petal po identyfikacji')

        plt.tight_layout()
        plt.show()


iris = load_iris()

data = pd.DataFrame(iris.data, columns=iris.feature_names)

knn_class = KnnClass(data)
knn_class2 = KnnClass(data)
knn_class3 = KnnClass(data)
euclides = knn_class.euclides_distance_func()
cosine = knn_class2.cosine_distance_func()
manhattan = knn_class3.manhattan_distance_func()

print(f"K rowna sie {knn_class.k}")

knn_class.plot_knn(euclides)
print(knn_class.accuracy, "%")

knn_class2.plot_knn(cosine)
print(knn_class2.accuracy, "%")

knn_class3.plot_knn(manhattan)
print(knn_class3.accuracy, "%")
