#!/usr/bin/env python
# coding: utf-8

# ### Нейросетевая модель прогнозирования температуры звёзд по астрофизическим характеристикам  
# 
# ## Описание проекта  
# 
# **Цель:** Разработка нейросетевой модели для точного предсказания температуры на поверхности звёзд, альтернативного традиционным физическим методам.  
# 
# **Традиционные методы:**  
# - Закон смещения Вина  
# - Закон Стефана-Больцмана  
# - Спектральный анализ  
# 
# **Преимущества ML-подхода:**  
# - Учёт комплексных взаимосвязей параметров  
# - Автоматизация расчётов  
# - Потенциально более высокая точность  
# 
# ## Данные обсерватории (240 звёзд)  
# 
# **Характеристики:**  
# - `Относительная светимость (L/Lo)`  
# - `Относительный радиус (R/Ro)`  
# - `Абсолютная звёздная величина (Mv)`  
# - `Цвет` (white, red, blue и др.)  
# - `Тип звезды` (0-5):  
#   - 0: коричневый карлик
#   - 1: красный карлик
#   - 2: белый карлик
#   - 3: звезды главной последовательности
#   - 4: сверхгигант
#   - 5: гипергигант 
# 
# **Целевая переменная:**  
# - `Абсолютная температура T(K)`  
# 
# ## Техническая реализация  
# 
# **Стек технологий:**  
# - Python (TensorFlow/Keras или PyTorch)  
# - Pandas/NumPy для обработки данных  
# - Matplotlib/Seaborn для визуализации  
# - Scikit-learn для предобработки  
# 
# **Архитектура модели:**  
# - Полносвязная нейронная сеть (DNN)  
# - Возможность использования:  
#   - Категориальных embedding'ов для цвета  
#   - Нормализации числовых признаков  
# 
# **Метрики оценки:**  
# - MAE (Mean Absolute Error)  
# - R²-score  
# - Сравнение с физическими методами  
# 
# > **Примечание:** Проект демонстрирует применение deep learning в астрофизике, заменяя классические физические расчёты.  

# ## Загрузка данных

# In[1]:


get_ipython().system('pip install -U scikit-learn')

# Стандартные библиотеки
import os
import warnings

# Научные вычисления и обработка данных
import pandas as pd
import numpy as np

# Графическая визуализация
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil

# Дополнительные библиотеки (установка через pip, если требуется)
try:
    import phik
    from phik.report import plot_correlation_matrix
except ImportError:
    get_ipython().system('pip install phik')
    import phik
    from phik.report import plot_correlation_matrix
    
# PyTorch для машинного обучения
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Scikit-Learn для работы с данными и метриками
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

# Проверка установленных библиотек
try:
    import pyparsing
except ImportError:
    get_ipython().system('pip install pyparsing')


# In[2]:


def summarize_dataframe(df):
    print('='*40)
    print(f'Общие размеры DataFrame: {df.shape[0]} строк, {df.shape[1]} столбцов')
    print('='*40)
    
    print('\nПервые 10 строк:')
    display(df.head(10))
    
    print('\nСтатистика числовых столбцов:')
    display(df.describe())
    
    print('\nИнформация о DataFrame:')
    info = df.info(memory_usage='deep')
    print('\nИспользование памяти: {:.2f} MB'.format(
        df.memory_usage(deep=True).sum() / (1024 ** 2)
    ))
    print('='*40)
    
    return info


# In[3]:


def visualize_data_spread(df, columns):
    # Проверка, является ли df объектом DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Первый аргумент должен быть объектом pandas.DataFrame.")
    
    for column in columns:
        # Проверка наличия столбца в DataFrame
        if column not in df.columns:
            print(f"Столбец '{column}' отсутствует в DataFrame.")
            print('-' * 40)
            continue

        # Диаграмма размаха
        plt.figure(figsize=(10, 6))
        df.boxplot(column=column)
        plt.title(f'Диаграмма размаха для столбца: {column}')
        plt.ylabel('Значения')
        plt.grid(True)
        plt.show()

        # Гистограмма
        plt.figure(figsize=(10, 6))
        df[column].plot(kind='hist', bins=30, grid=True, color='skyblue', edgecolor='black')
        plt.title(f'Гистограмма для столбца: {column}')
        plt.xlabel(column)
        plt.ylabel('Количество наблюдений')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        print('-' * 40)


# In[4]:


def analyze_uniqueness(df, columns):
    for column in columns:
        # Проверка наличия столбца в DataFrame
        if column not in df.columns:
            print(f"Столбец '{column}' отсутствует в DataFrame.")
            print('-' * 40)
            continue

        # Анализ уникальных значений
        unique_values = df[column].unique()
        num_unique = df[column].nunique()

        print(f"Столбец: {column}")
        print(f"Количество уникальных значений: {num_unique}")
        if num_unique <= 20:
            print(f"Уникальные значения: {unique_values}")
        else:
            print(f"Уникальные значения (первые 20): {unique_values[:20]} ...")
        print('-' * 40)


# In[5]:


def fact_forecast(test_preds, y_test):   
    y1 = torch.FloatTensor(test_preds)
    y1 = y1.detach().numpy().reshape([-1])
    y2 = y_test.detach().numpy().reshape([-1])
    x = np.arange(len(y1))

    fig,ax = plt.subplots()
    fig.set_figwidth(18)
    fig.set_figheight(8)
    ax.set_xticks(x)
    fact = ax.bar(x, y2, width = 0.6, label = 'Факт')
    forecast = ax.bar(x, y1, width = 0.3, label = 'Прогноз')
    ax.legend()
    ax.set_title('График "Факт-прогноз"', fontsize=20)
    ax.set_xlabel('Номер звезды')
    ax.set_ylabel('Температура звезды')
    plt.show()


# In[6]:


dataset = '/datasets/6_class.csv'

if os.path.exists(dataset):
    data = pd.read_csv(dataset, sep=',', index_col = 0)
else:
    print('Something is wrong')
summarize_dataframe(data)


# In[7]:


# # Посмотрим на разброс данных
visualize_data_spread(data, ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star type'])


# In[8]:


# Проверим корректность категориальных данных
analyze_uniqueness(data, ['Star type', 'Star color'])


# In[9]:


# Проверка корреляционной матрицы для выявления зависимости между переменными
print("Корреляционная матрица:")
#corr_matrix = data.corr()
#display(corr_matrix)
print("=" * 50)

# Проверка выбросов для столбца "Temperature (K)"
temperature_outliers = data.loc[data['Temperature (K)'] > 32000].sort_values(by='Temperature (K)')
if not temperature_outliers.empty:
    print(f"Найдены выбросы в столбце 'Temperature (K)' (порог > 32000):")
    display(temperature_outliers)
else:
    print("Выбросов в столбце 'Temperature (K)' не обнаружено.")
print("=" * 50)

# Проверка выбросов для столбца "Luminosity(L/Lo)"
luminosity_outliers = data.loc[data['Luminosity(L/Lo)'] > 100000].sort_values(by='Luminosity(L/Lo)')
if not luminosity_outliers.empty:
    print(f"Найдены выбросы в столбце 'Luminosity(L/Lo)' (порог > 100000):")
    display(luminosity_outliers)
else:
    print("Выбросов в столбце 'Luminosity(L/Lo)' не обнаружено.")
print("=" * 50)

# Проверка выбросов для столбца "Radius(R/Ro)"
radius_outliers = data.loc[data['Radius(R/Ro)'] > 500].sort_values(by='Radius(R/Ro)')
if not radius_outliers.empty:
    print(f"Найдены выбросы в столбце 'Radius(R/Ro)' (порог > 500):")
    display(radius_outliers)
else:
    print("Выбросов в столбце 'Radius(R/Ro)' не обнаружено.")
print("=" * 50)


# ### Вывод
# 
# Предоставленный нам датасет содержит 240 записей, описанных двумя категориальными (Star type, Star color) и четырьмя числовыми (Temperature (K), Luminosity (L/Lo), Radius (R/Ro), Absolute magnitude (Mv)) признаками. Наша цель — разработать модель для прогнозирования Temperature (K), поэтому важно было проверить данные на ошибки и аномалии. В каждом из признаков были обнаружены выбросы, и мы подробно рассмотрим их характеристики:
# 
# - Temperature (K) — медианное значение температуры в выборке превышает 5000 К, в то время как большая часть наблюдений сосредоточена в диапазоне 3500–4000 Кельвинов, что соответствует средней температуре большинства звёзд. Однако встречаются экстремальные значения (>32500 К), которые мы удалять не будем, поскольку это целевой признак, и подобные значения соответствуют реальным горячим звёздам. Например, известно, что температура звёзд варьируется от 2000 до 80000 К в зависимости от их типа.
# 
# - Luminosity (L/Lo) — светимость большинства звёзд в датасете близка к нулю, но есть редкие наблюдения с уровнями от 100,000 и выше. Это соответствует реальным данным, поскольку в зависимости от радиуса и температуры светимость может сильно отличаться. В нашей Вселенной существуют звёзды, светимость которых превышает солнечную в 8.1 миллиона раз. Выбросы, наблюдаемые в этом признаке, тесно связаны с другими величинами, такими как радиус и температура, поэтому их также не будем удалять.
# 
# - Radius (R/Ro) — аналогично предыдущему признаку, большинство наблюдений имеют радиусы меньше солнечного (медиана около 0), но встречаются звёзды, радиус которых превышает солнечный в тысячи раз. Эти объекты принадлежат к классу гипергигантов (5), и такие значения также останутся в данных.
# 
# - Absolute magnitude (Mv) — данный признак отличается бимодальным распределением, симметричным относительно нуля. В диапазоне от 0 до 5 наблюдается наименьшее количество значений, а выбросов в этом признаке нет.
# 
# - Star type — категориальный столбец с равномерным распределением классов (по 40 записей каждого типа).
# 
# Перейдём к общему качеству данных. Столбцы корректно названы, типы данных соответствуют их значениям, отсутствуют пропуски. Однако в столбце Star color обнаружены неявные дубликаты: различия в регистре (например, "blue" и "Blue") и несогласованные форматы написания (например, "blue-white" и "white-blue"). Эти несоответствия необходимо будет исправить на этапе предобработки. Подобные проблемы часто возникают из-за человеческого фактора, когда разные записи вносятся вручную.

# ## Предобработка и анализ данных

# In[11]:


print("Обработка неявных дубликатов и классификация цветов в столбце 'Star color'...")

# Приведение значений к единому формату (нижний регистр, удаление пробелов)
data['Star color'] = data['Star color'].str.lower().str.strip()

# Словарь для замены и унификации
color_mapping = {
    'red': 'red',
    'blue white': 'blue-white',
    'blue-white': 'blue-white',
    'blue': 'blue',
    'white': 'white',
    'yellowish white': 'white-yellow',
    'yellow white': 'white-yellow',
    'white-yellow': 'white-yellow',
    'yellowish': 'yellow',
    'orange': 'orange',
    'orange-red': 'other',
    'pale yellow orange': 'other',
    'whitish': 'other',
}

# Применение маппинга для унификации
data['Star color'] = data['Star color'].replace(color_mapping)

# Дополнительная проверка и замена редких значений на 'other'
valid_colors = {'blue', 'blue-white', 'white', 'white-yellow', 'yellow', 'orange', 'red'}
data['Star color'] = data['Star color'].apply(lambda x: x if x in valid_colors else 'other')

# Проверка уникальных значений после обработки
print("Уникальные значения после обработки:")
unique_colors = data['Star color'].unique()
print(f"Столбец 'Star color' — {len(unique_colors)} уникальных значений:")
print(", ".join(map(str, unique_colors)))


# In[12]:


print(f'Количество явны дубликатов = {data.duplicated().sum()}')


# После выполнения предобработки данных были устранены проблемы в столбце Star color, связанные с неявными дубликатами слов. Это позволило уменьшить количество уникальных значений с 19 до 8. Явные дубликаты в данных отсутствуют.

# In[13]:


# Вычисление и визуализация корреляции с использованием Phik
print("Вычисляем матрицу корреляции Phi_k...")

# Расчёт матрицы корреляции Phi_k
phik_matrix = data.phik_matrix()
phik_matrix_rounded = phik_matrix.round(2)  # Округляем значения для удобства анализа

# Вывод первых строк матрицы корреляции
print("Матрица корреляции Phi_k (округлена до двух знаков):")
display(phik_matrix_rounded)

# Визуализация матрицы корреляции
print("Визуализируем матрицу корреляции Phi_k...")
plot_correlation_matrix(
    phik_matrix.values, 
    x_labels=phik_matrix.columns, 
    y_labels=phik_matrix.index, 
    vmin=0, vmax=1, color_map="coolwarm", 
    title=r"Матрица корреляции $\phi_K$", 
    fontsize_factor=1.2, 
    figsize=(12, 10)
)

# Оптимизация отображения графика
plt.tight_layout()
plt.show()


# In[14]:


significance_overview = data.significance_matrix(interval_cols=['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star type'])
plot_correlation_matrix(significance_overview.fillna(0).values, 
                        x_labels=significance_overview.columns, 
                        y_labels=significance_overview.index, 
                        vmin=-5, vmax=5, title="Статистическая значимость признаков", 
                        usetex=False, fontsize_factor=1.5, figsize=(10, 7), color_map="Blues")
plt.tight_layout()


# In[15]:


# Построение тепловой карты для корреляции с таргетом
plt.figure(figsize=(5, 8))
sorted_phik = phik_matrix.sort_values(by='Temperature (K)', ascending=False)[['Temperature (K)']]

# Настройка тепловой карты
sns.heatmap(
    sorted_phik,
    cmap='Blues',
    annot=True,
    annot_kws={'size': 16, 'weight': 'bold'},
    fmt='.2g',
    cbar_kws={'label': 'Correlation'},
    linewidths=1,
    linecolor='black'
)

# Добавляем заголовок
plt.title('Корреляция с температурой (Таргет)', fontsize=22, fontweight='bold')

# Отображаем график
plt.show()


# In[16]:


# Посмотрим на глобальную корреляцию признаков
global_correlation, global_labels = data.global_phik(
    interval_cols=[
        'Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 
        'Absolute magnitude(Mv)', 'Star type'
    ]
)

# Настройка визуализации с улучшенным внешним видом
plt.figure(figsize=(6, 8))

# Построение тепловой карты с градиентом от темного синего к белому и черной рамкой между клетками
sns.heatmap(
    global_correlation,
    cmap='Blues',
    vmin=0, vmax=1,
    annot=True,
    annot_kws={'size': 14, 'weight': 'bold', 'color': 'black'},
    linewidths=2,
    linecolor='black',
    xticklabels=[''],
    yticklabels=global_labels,
)

# Заголовок с формулой LaTeX
plt.title(r"Global $g_k$ Correlation", fontsize=22, fontweight='bold')

# Оптимизация расположения элементов графика
plt.tight_layout()

# Отображаем график
plt.show()


# In[17]:


features = ['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
hue = 'Star type'

# Установка стиля
sns.set(style="whitegrid")

# Построение графиков
for feature in features:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=feature, y='Star type', hue=hue, palette='tab10')
    plt.title(f"Диаграмма рассеяния: {feature} vs {hue}")
    plt.xlabel(feature)
    plt.ylabel('Star type')
    plt.legend(title='Star type')
    plt.show()


# ### Вывод
# 
# Первое, на что стоит обратить внимание, — это явная мультикорреляция между столбцами Star type и Absolute magnitude (Mv), составляющая 92% по корреляции phik. Несмотря на это, мы не будем исключать ни один из этих признаков, поскольку их статистическая значимость составляет 21,6, что достаточно высоко. Теперь давайте обратим внимание на взаимосвязь таргета с остальными признаками:
# 
# - 'Luminosity (L/Lo)' — 50% корреляция. Часто можно встретить мнение, что температура звезды определяется её цветом, однако светимость звезды относительно Солнца дает нам точное представление о её температуре, хотя и не напрямую.
# 
# - 'Absolute magnitude (Mv)' — 69% корреляция. Это физическая величина, характеризующая яркость астрономического объекта, если бы он находился на стандартном расстоянии от наблюдателя. Абсолютная звёздная величина дает возможность сравнивать реальную, а не наблюдаемую светимость, что делает её аналогичной понятию относительной светимости.
# 
# - Star color — 70% корреляция. Как уже упоминалось, цвет звезды является одним из основных индикаторов её температуры, что, скорее всего, связано с изменением пути светового потока при достижении Земли.
# 
# - Star type — 60% корреляция. Поскольку звезды делятся на пять основных типов, можно предположить их температуру, основываясь на типе, с учетом статистических данных и наблюдений.
# 
# - 'Radius (R/Ro)' — самая слабая корреляция, всего 24%. Это может быть связано с природой звёзд, а именно с их плотностью и массой. Размеры объекта сами по себе не так важны, как то, из чего он состоит и какой световой спектр испускает.
# 
# Тип звезды существенно влияет на её физические характеристики:
# 
# - Горячие звёзды (3, 4, 5 типов) выделяются высокой температурой и яркостью.
# - Холодные звёзды (0, 1, 2 типов) обладают более низкими температурами и светимостью, но их блеск в абсолютных величинах может быть выше.
# - Крупные звёзды (4, 5 типов) значительно больше Солнца, в то время как звёзды 0–3 типов сравнительно малы.
# 
# Эти различия подчёркивают разнообразие звёзд по физическим характеристикам и показывают корреляцию между их температурой, светимостью, радиусом и абсолютной звёздной величиной.

# In[18]:


# Включим вывод предупреждений
warnings.filterwarnings('default')


# In[20]:


# Поставим ограничений на число столбцов
pd.set_option('display.max_rows', 10)


# In[22]:


X = data.drop('Temperature (K)', axis=1)
y = data['Temperature (K)']

# Разделение на обучающую и тестовую выборки с стратификацией по цвету звезд
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=X['Star color']
)

# Определение числовых и категориальных признаков
numerics = ['Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)']
categorical = ['Star color', 'Star type']

# Создание ColumnTransformer
col_transformer = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), numerics),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False), categorical)
    ],
    remainder="passthrough"  # Оставить остальные столбцы без изменений
)

# Подгонка трансформера на обучающих данных
col_transformer.fit(X_train)

# Трансформация обучающих и тестовых данных
X_train = col_transformer.transform(X_train)
X_test = col_transformer.transform(X_test)


# ## Построение базовой нейронной сети

# In[23]:


X_train = torch.FloatTensor(np.array(X_train))
X_test = torch.FloatTensor(np.array(X_test))
y_train = torch.FloatTensor(np.array(y_train))
y_test = torch.FloatTensor(np.array(y_test))


# In[24]:


class CustomNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super(CustomNet, self).__init__()
        
        self.layer1 = nn.Linear(input_size, hidden_size_1)
        self.relu1 = nn.ReLU()        
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.relu2 = nn.ReLU()        
        self.layer3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)        
        x = self.layer2(x)
        x = self.relu2(x)        
        x = self.layer3(x)
        
        return x


# In[25]:


input_neurons = X_train.shape[1]
hidden_neurons_1 = 15
hidden_neurons_2 = 10
output_neurons = 1

net = CustomNet(input_neurons, hidden_neurons_1, hidden_neurons_2, output_neurons)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

loss = nn.MSELoss()

print(f"Модель с {input_neurons} входами, {hidden_neurons_1} нейронами в первом скрытом слое, "
      f"{hidden_neurons_2} нейронами во втором скрытом слое и {output_neurons} выходами.")


# In[26]:


dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
dataset_test = torch.utils.data.TensorDataset(X_test, y_test)

train_dataloader = DataLoader(dataset_train, batch_size=40, shuffle=True, num_workers=0)

test_dataloader = DataLoader(dataset_test, batch_size=40, num_workers=0)

print(f"Тренировочный DataLoader: {len(train_dataloader)} батчей, тестовый DataLoader: {len(test_dataloader)} батчей")


# In[27]:


class EarlyStoppingCallback():
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


# In[28]:


early_stopping = EarlyStoppingCallback(patience=5, min_delta=20)

num_epochs = 100001
for epoch in range(num_epochs):
    net.train()
    for batch in train_dataloader:
        data_train, temperature_train = batch
        optimizer.zero_grad()

        preds = net.forward(data_train).flatten()

        loss_value = torch.sqrt(loss(preds, temperature_train))
        loss_value.backward()
        optimizer.step()
    
    if epoch % 200 == 0:
        predicted_temp = [] 
        with torch.no_grad():
            net.eval()
            for batch in test_dataloader:
                data_test, temperature_test = batch

                test_preds = net.forward(data_test).flatten()
                predicted_temp.append(test_preds)
                RMSE_loss = (loss(test_preds, temperature_test)**0.5)

        predicted_temp = torch.cat(predicted_temp).detach().numpy()
        RMSE = mean_squared_error(y_test, predicted_temp)**0.5
        early_stopping(RMSE)
        if early_stopping.counter == 0:
            best_rmse = RMSE
            best_predicted_temp = predicted_temp
        print(f"epoch:{epoch}, RMSE test: {RMSE}")

        if early_stopping.early_stop:
            print('Early Stoppning!!!')
            print(f'Best RMSE test {best_rmse}')
            break 


# In[29]:


# Визуализируем полученные значения
fact_forecast(predicted_temp, y_test)


# ### Вывод
# 
# Лучшая метрика RMSE до начала переобучения равна 4422. Это не лучший результат, которого мы можем добиться, далее будем улучшаять нашу НС. НС просто не смогла правильно научиться на таком малом количестве данных. 

# ## Улучшение нейронной сети

# In[30]:


input_neurons = X_train.shape[1]
hidden_neurons_1 = 15
hidden_neurons_2 = 10
output_neurons = 1

# Создаем модель нейросети с заданной архитектурой
net = CustomNet(input_neurons, hidden_neurons_1, hidden_neurons_2, output_neurons)

# Настроим несколько оптимизаторов для выбора на основе разных методов оптимизации
optimizers = {
    'Adam': torch.optim.Adam(net.parameters(), lr=1e-3),
    'NAdam': torch.optim.NAdam(net.parameters(), lr=1e-2),
    'Adamax_1': torch.optim.Adamax(net.parameters(), lr=1e-2),
    'Adamax_2': torch.optim.Adamax(net.parameters(), lr=1e-3),
    'Adam_2': torch.optim.Adam(net.parameters(), lr=1e-2)
}

# Выбор оптимизатора, например, по ключу
chosen_optimizer = optimizers['Adam']

# Выводим информацию о выбранном оптимизаторе
print(f"Используется оптимизатор: {chosen_optimizer.__class__.__name__} с lr={chosen_optimizer.param_groups[0]['lr']}")


# In[31]:


rmse_optimizers = []

# Перебираем оптимизаторы (значения словаря, а не ключи)
for optimizer_name, optimizer in optimizers.items():
    print(f"Используем оптимизатор: {optimizer_name}")
    print()

    early_stopping = EarlyStoppingCallback(patience=5, min_delta=20)

    num_epochs = 100001
    for epoch in range(num_epochs):
        net.train()
        for batch in train_dataloader:
            data_train, temperature_train = batch
            optimizer.zero_grad()

            preds = net.forward(data_train).flatten()

            loss_value = (loss(preds, temperature_train)**0.5)
            loss_value.backward()
            optimizer.step()

        if epoch % 200 == 0:
            predicted_temp = []
            with torch.no_grad():
                net.eval()
                for batch in test_dataloader:
                    data_test, temperature_test = batch

                    test_preds = net.forward(data_test).flatten()
                    predicted_temp.append(test_preds)
                    RMSE_loss = torch.sqrt(loss(test_preds, temperature_test))

            predicted_temp = torch.cat(predicted_temp).detach().numpy()
            RMSE = mean_squared_error(y_test, predicted_temp)**0.5
            early_stopping(RMSE)

            if early_stopping.counter == 0:
                best_rmse = RMSE
                best_predicted_temp = predicted_temp
            print(f"epoch:{epoch}, RMSE test: {RMSE}")

            if early_stopping.early_stop:
                print('Early Stoppning!!!')
                print(f'Best RMSE test {best_rmse}')
                rmse_optimizers.append(round(best_rmse, 2))
                break


# In[32]:


optimizers_rmse = pd.DataFrame(data = rmse_optimizers, index = ['Adam, lr: 0.001',
                                                                'NAdam, lr: 0.01',
                                                                'Adamax, lr: 0.01',
                                                                'Adamax, lr: 0.001',
                                                                'Adam, lr: 0.01'], columns = ['RMSE'])
display(optimizers_rmse.sort_values(by='RMSE'))


# In[33]:


class EnhancedNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.5):
        super(EnhancedNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.act1 = nn.ReLU()
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.act2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
        x = self.fc3(x)
        return x


# In[34]:


input_neurons = X_train.shape[1]
hidden_neurons_1 = 15
hidden_neurons_2 = 10
output_neurons = 1

net = EnhancedNet(input_neurons, hidden_neurons_1, hidden_neurons_2, output_neurons)
optimizer = torch.optim.NAdam(net.parameters(), lr=1e-2)
loss = nn.MSELoss()


# In[35]:


dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
dataset_test = torch.utils.data.TensorDataset(X_test, y_test)

train_dataloader = DataLoader(dataset_train, batch_size=40, shuffle=True,
                              num_workers=0)
test_dataloader = DataLoader(dataset_test, batch_size=40, num_workers=0)  


# In[36]:


early_stopping = EarlyStoppingCallback(patience=5, min_delta=20)

num_epochs = 100001
val_loss = []
train_loss = []

for epoch in range(num_epochs):
    net.train()
    train_loss_batches = []
    for batch in train_dataloader:
        data_train, temperature_train = batch
        optimizer.zero_grad()

        preds = net.forward(data_train).flatten()

        loss_value = (loss(preds, temperature_train)**0.5)
        
        loss_value.backward()
        optimizer.step()
        
        loss_value = loss_value.detach().numpy().reshape([-1])
        train_loss_batches.append(loss_value)
    
        
    if epoch % 200 == 0:
        predicted_temp = [] 
        with torch.no_grad():
            net.eval()
            for batch in test_dataloader:
                data_test, temperature_test = batch

                test_preds = net.forward(data_test).flatten()
                predicted_temp.append(test_preds)
                RMSE_loss = torch.sqrt(loss(test_preds, temperature_test))

        predicted_temp = torch.cat(predicted_temp).detach().numpy()
        RMSE = mean_squared_error(y_test, predicted_temp)**0.5
        
        
        early_stopping(RMSE)
        if early_stopping.counter == 0:
            best_rmse = RMSE
            best_predicted_temp = predicted_temp
            val_loss.append(best_rmse)
            train_loss.append(np.mean(train_loss_batches))
        print(f"epoch:{epoch}, RMSE test: {RMSE}")
        
        
        if early_stopping.early_stop:
            print('Early Stoppning!!!')
            print(f'Best RMSE test {best_rmse}')
            break 


# In[37]:


x = np.arange(len(val_loss))

fig, ax = plt.subplots()
fig.set_figwidth(10)
fig.set_figheight(7)
ax.set_xticks(x)
val = ax.plot(x, val_loss, label = 'Validation loss')
train = ax.plot(x, train_loss, label = 'Train loss')
ax.legend()
ax.set_title('Потери нейросети', fontsize=20)
ax.set_xlabel('Эпохи (600)')
ax.set_ylabel('Loss')
plt.show()


# In[38]:


fact_forecast(predicted_temp, y_test)


# ## Выводы

# Мы провели перебор оптимизаторов, чтобы выявить лучший из пяти представленных, учитывая только конечную метрику RMSE, без учёта скорости обучения. После анализа, мы пришли к выводу, что наилучшим выбором оказался NAdam с шагом 0.01, который продемонстрировал RMSE равную 4336.
# 
# На этом этапе мы внедрили дополнительные улучшения в модель, добавив Dropout на 50% и регуляризацию весов с помощью BatchNorm1d для первого скрытого слоя. Хотя это увеличило время обучения, оно способствовало значительному улучшению метрики. В итоге, конечная метрика RMSE для усовершенствованной нейросети составила 4263.
# 
# На графике потерь нашей последней модели можно заметить, что она обучалась эффективно, без недо- или переобучения, благодаря использованию ранней остановки.

# In[ ]:




