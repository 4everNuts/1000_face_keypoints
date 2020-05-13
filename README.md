### Решение контеста https://www.kaggle.com/c/made-thousand-facial-landmarks по поиску ключевых точек лица

Как запускать:
0) **Установка пакетов:**   
pip install -r requirements.txt
1) **Создание датасетов при каждом запуске скрипта занимало 20 минут, так что сохраняем датасеты в один файл:**  
 ```python save_datasets.py --data "PATH_TO_DATA"```
2) **Поиск гиперпараметров:**  
Запускаем следующий скрипт много раз с разными параметрами, пытаемся минимизировать валидационный лосс (оптимайзер тоже следовало бы добавить в параметры, но руки не дошли, так что менял его в коде) на исходном train-val разбиении:   
```python hack_train.py --name=resnext50_32x4d --data="PATH_TO_DATA" -e=30 --gpu --batch-size=512 -lr=0.001```   
Результаты можно посмотреть в stats.xlsx
3) **Тренировка фолдов:**  
Когда нашли оптимальный сетап, объединяем train и val датасеты, и делим на 5 фолдов. Это делается добавлением параметров **fold-prefix** (номер эксперимента) и **fold** (номер фолда):  
```python hack_train.py --name=resnext50_32x4d --data="PATH_TO_DATA" -e=30 --gpu --batch-size=512 -lr=0.001 --fold=0 --fold-prefix=0```  
Затем усредняем предсказания кодом из Create_folds_submission.ipynb, часть Averaging prediction
