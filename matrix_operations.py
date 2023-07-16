import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib_inline
import math

def transpozycja(macierz):
    row = len(macierz)
    col = len(macierz[0])

    transponowana = [[0 for j in range(row)] for i in range(col)]

    for i in range(row):
        for j in range(col):
            transponowana[j][i] = macierz[i][j]

    return transponowana


#print(transpozycja([[1, 2, 3], [4, 5, 6]]))

def mnozenie(macierz1, macierz2):
    przemnozone = [[0 for row in range(len(macierz1))] for col in range(len(macierz2[0]))]
    if len(macierz1) != len(macierz2):
        return print("error, niezgodnosc wymiarow")
    for i in range(len(macierz1)):
        for j in range(len(macierz2[0])):
            for k in range(len(macierz2)):
                przemnozone[i][j] += macierz1[i][k] * macierz2[k][j]
    return przemnozone


#print(mnozenie([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def wyznacznik(macierz):
    if len(macierz) == 2:
        return macierz[0][0] * macierz[1][1] - macierz[0][1] * macierz[1][0]
    if len(macierz) == 3:
        a, b, c = macierz[0]
        d, e, f = macierz[1]
        g, h, i = macierz[2]
        return a * e * i + b * f * g + c * d * h - c * e * g - b * d * i - a * f * h
    else:
        return print('nieprawidlowe dane')


#print(wyznacznik([[1, 2, 3], [4, 5, 6], [7, 8, 12]]))


def minor(macierz, i, j):
    rows = [row[:j] + row[j+1:] for row in macierz[:i] + macierz[i+1:]]
    return rows


def dopelnienia(macierz):
    macierz_dopelnien = []
    det = wyznacznik(macierz)
    if det != 0:
        for i in range(len(macierz)):
            row = []
            for j in range(len(macierz)):
                minor_ij = minor(macierz, i, j)
                minor_wyznacznik = wyznacznik(minor_ij)
                minor_dopelnienie = minor_wyznacznik * (-1) ** (i + j)
                row.append(minor_dopelnienie)
            macierz_dopelnien.append(row)
        macierz_dopelnien = transpozycja(macierz_dopelnien)
        for i in range(len(macierz)):
            for j in range(len(macierz)):
                macierz_dopelnien[i][j] = macierz_dopelnien[i][j] / det
        return macierz_dopelnien
    else:
        print("wyznacznik 0")


#print(dopelnienia([[1, 2, 3], [0, 1, 1], [2, 1, 2]]))


def odwrotna_gauss_new_but_old(macierz):
    rozmiar = len(macierz)
    macierz_rozszerzona = [row + [0] * rozmiar for row in macierz]
    for i in range(rozmiar):
        macierz_rozszerzona[i][i+rozmiar] = 1
    for i in range(rozmiar):
        max_row = max(range(i, rozmiar), key=lambda r: abs(macierz_rozszerzona[r][i]))
        macierz_rozszerzona[i], macierz_rozszerzona[max_row] = macierz_rozszerzona[max_row], macierz_rozszerzona[i]
        skalowanie = macierz_rozszerzona[i][i]
        if skalowanie == 0:
            return print("Macierz jest osobliwa i nie ma odwrotności")
        for j in range(i, rozmiar * 2):
            macierz_rozszerzona[i][j] /= skalowanie
        for row in range(rozmiar):
            if row == i:
                continue
            mnoznik = macierz_rozszerzona[row][i]
            for col in range(i, rozmiar * 2):
                macierz_rozszerzona[row][col] -= mnoznik * macierz_rozszerzona[i][col]
    return [[macierz_rozszerzona[row][col] for col in range(rozmiar, rozmiar * 2)] for row in range(rozmiar)]


#print(odwrotna_gauss_new_but_old([[1, 1, 2, 2, 3], [1, 2, 2, 2, 2], [1, 1, 2, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 1, 2]]))
#print(odwrotna_gauss_new_but_old([[2, 3, 5, 1, 7, 11], [7, 1, 8, 2, 9, 4], [4, 6, 1, 3, 5, 2], [2, 5, 9, 4, 1, 8], [3, 4, 6, 7, 8, 1], [9, 1, 4, 5, 3, 6]]))
#print(odwrotna_gauss_new_but_old([[1, 2, 3, 4, 5, 6],[2, 4, 6, 8, 10, 12],[3, 6, 9, 12, 15, 18],[4, 8, 12, 16, 20, 24],[5, 10, 15, 20, 25, 30],[6, 12, 18, 24, 30, 36]]))

def uklad(a1, b1, c1, a2, b2, c2):
    d = wyznacznik([[a1, b1], [a2, b2]])
    dx = wyznacznik([[c1, b1], [c2, b2]])
    dy = wyznacznik([[a1, c1], [a2, c2]])
    if d != 0:
        x = dx / d
        y = dy / d
    else:
        if dx != 0 and dy != 0:
            print('uklad sprzeczny')
            return 0
        else:
            if dx == 0 or dy == 0:
                print('nieskonczenie wiele rozwiazan')
                return 0
    return x, y


#print(uklad(3, 2, 5, 4, 7, -2))
#print(uklad(2, -1, 2, 1, -0.5, 1))
#print(uklad(1, -3, 2, 1, -3, 5))


def rzad_macierzy(macierz):
    rozmiar = len(macierz)
    macierz_rozszerzona = [row[:] for row in macierz]
    rzad = 0
    for i in range(rozmiar):
        wiersz = None
        for j in range(i, rozmiar):
            if any(macierz_rozszerzona[j][k] != 0 for k in range(i, rozmiar)):
                wiersz = j
                break
        if wiersz is None:
            return rzad
        if wiersz != i:
            macierz_rozszerzona[i], macierz_rozszerzona[wiersz] = macierz_rozszerzona[wiersz], macierz_rozszerzona[i]
        for j in range(i+1, rozmiar):
            if macierz_rozszerzona[j][i] != 0:
                mnoznik = macierz_rozszerzona[j][i] / macierz_rozszerzona[i][i] if macierz_rozszerzona[i][i] != 0 else 0
                for k in range(i, rozmiar):
                    macierz_rozszerzona[j][k] -= mnoznik * macierz_rozszerzona[i][k]
        rzad += 1
    return rzad

#print(rzad_macierzy([[1, 2, 3], [0, 1, 1], [2, 1, 2]]))
#print(rzad_macierzy([[1, 1, 2, 2, 3], [1, 2, 2, 2, 2], [1, 1, 2, 2, 2], [1, 1, 1, 2, 2], [1, 1, 1, 1, 2]]))
#print(rzad_macierzy([[1, 2, 3, 4, 5, 6],[2, 4, 6, 8, 10, 12],[3, 6, 9, 12, 15, 18],[4, 8, 12, 16, 20, 24],[5, 10, 15, 20, 25, 30],[6, 12, 18, 24, 30, 36]]))



df = pd.DataFrame()
df['X'] = [1, 2, 3, 4, 5]
df['Y'] = [4, 6, 9, 11, 18]
print(df)
plt.scatter(df['X'], df['Y'], label='Wartosci')
plt.xlabel('Wartosci X')
plt.ylabel('Wartosci Y')
plt.legend()
plt.show()

n = len(df['X'])
pearson = pd.DataFrame(df[:])
pearson['y2'] = df['Y'] * df['Y']
pearson['xy'] = df['X'] * df['Y']
pearson['x2'] = df['X'] * df['X']
pearson.loc['sum'] = pearson.sum()
print()
print(pearson)


def srednia(zbior):
    return float(sum(zbior) / len(zbior))


def odchylenie(zbior):
    wynik = 0
    for i in range(len(zbior)):
        wynik += (zbior[i] - srednia(zbior)) ** 2
    return math.sqrt(wynik / len(zbior - 1))


def pearson_kor(t, n):
    return ((len(df['X'])*t['xy']['sum']) - (t['X']['sum'] * t['Y']['sum'])) / math.sqrt(((len(df['X'])*t['x2']['sum']) - (t['X']['sum'] ** 2)) * ((len(df['X'])*t['y2']['sum']) - (t['Y']['sum'] ** 2)))



Sx = odchylenie(df['X'])
Sy = odchylenie(df['Y'])
#print("Odchylenie standardowe x: ", Sx)
#print("Odchylenie standardowe y: ", Sy)


mean_x = srednia(df['X'])
mean_y = srednia(df['Y'])
#print("Srednia x:", mean_x)
#print("Srednia y:", mean_y)
kor = pearson_kor(pearson, n)
print("Wspol kor zbioru: ", kor)

b = kor * (Sy / Sx)
a = mean_y - (b * mean_x)

print("a jest rowne: ", a)
print("b jest rowne: ", b)


def linia_regresji(x):
    return (b * x) + a


x = np.linspace(0, 5, 1000)
plt.scatter(df['X'], df['Y'], label='Wartosci niezalezne')
plt.plot(x, linia_regresji(x), 'r', label='Linia regresji')
plt.xlabel('Wartosci X')
plt.ylabel('Wartosci Y')
plt.legend()
plt.show()

df = df._append({'X': 6, 'Y': np.nan}, ignore_index=True)

def predict_y(x, b, a):
    return b * x + a
df.at[5, 'Y'] = predict_y(df['X'][5], b, a)


print(df)

df = df._append({'X': 7, 'Y': np.nan}, ignore_index=True)
df.at[6, 'Y'] = predict_y(df['X'][6], b, a)

print(df)

df = df._append({'X': 8, 'Y': np.nan}, ignore_index=True)
df.at[7, 'Y'] = predict_y(df['X'][7], b, a)

print(df)