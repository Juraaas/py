#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt  
import pandas as pd
import numpy as np
from matplotlib import cm  
import seaborn as sns  
from typing import List, Tuple, Dict
import math


# In[13]:


df: pd.DataFrame = pd.read_csv('df.csv', sep ='@')
    
df


# In[23]:


#zad1
def get_column_name(df: pd.DataFrame, mean: float, std: float) -> str:
    df_num: pd.DataFrame = df.select_dtypes(include=['number'])
    li: List[str] = df_num.columns
    for i in range(len(li)):
        if mean == df[li[i]].mean().round(6) and std == df[li[i]].std().round(6):
            return li[i]
new_columns_names: Dict = {get_column_name(df, 2.308642, 0.836071): 'class',
                           get_column_name(df, 0.383838, 0.486592): 'survived',
                           get_column_name(df, 446.000000, 257.353842): 'passengerId',
                           get_column_name(df, 29.699118, 14.526497): 'age',
                           get_column_name(df, 32.204208, 49.693429): 'cost'}
df: pd.DataFrame = df.rename(columns=new_columns_names)
    
df


# In[31]:


#zad 3
podzial: pd.DataFrame = df.groupby('class').agg({'passengerId': 'count'})
podzial.plot.pie(subplots=True, autopct='%.1f %%', xlabel='', ylabel='', title='Wykres kołowy', figsize=(7, 7))

plt.show()


# In[43]:


#zad2
a: np.ndarray = df['cost'].values
b: np.ndarray = df['age'].values
class_values: np.ndarray = df['class'].values
colors: np.ndarray = np.empty((0,len(a)), str)


for i in range(len(a)):
    if class_values[i] == 1:
        colors = np.append(colors, 'green')
    elif class_values[i] == 2:
        colors = np.append(colors, 'red')
    else:
        colors = np.append(colors, 'blue')

plt.scatter(x='cost', y='age', c=colors, data=df)
plt.legend(title='')
plt.xlabel('Koszt')
plt.ylabel('Wiek')
plt.title('Wykres punktowy')

plt.show()


# In[34]:


#zad4
wykres = sns.countplot(x='survived', hue='class', data=df) 
wykres.set(xlabel='Przeżywalność pasażerów z podziałem na klasy', ylabel='Liczba Pasażerów', title='Wykres słupkowy')

plt.show()


# In[ ]:

def number_format(number: int, mask: str):


    wynik: str = ''
    tymczasowa: str = str(number)
    j: int = 0

    for i in range (0, len(mask)):
        if mask[i] == '#':
            wynik += tymczasowa[j]
            j += 1
        else:
            wynik += mask[i]
    return wynik

print(number_format(123456789,'##-##-##'))
print(number_format(123456789,'#,#,#,#,#,#,#'))

class Circle:
    def __init__(self, r: float):
        self.r = r
    def area(self):
        return math.pi * (self.r ** 2)

    def __sub__(self, other):
        if other.area() > self.area():
            return (other.area() - self.area())
        else:
            return (self.area() - other.area())




circ = Circle(3)
circ2 = Circle(5)





class Square:
    def __init__(self, side: float) -> None:
        self.side = side
    def area(self):
       return self.side * self.side

    def __sub__(self, other):
        if other.area() > self.area():
            return (other.area() - self.area())
        else:
            return (self.area() - other.area())



kwadrat = Square(3)
kwadrat2 = Square(7)


print('dla kola o promieniu 5 i kwadratu o promieniu 3', kwadrat.__sub__(circ2))
print('dla kwadratu o promieniu 7 i kola o promieniu 3', circ.__sub__(kwadrat2))


