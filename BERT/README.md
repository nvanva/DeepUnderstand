# Запуск визуализатора
```
python main.py -p [port_number]
```

# Обработать датасет
Требуется чекпоинт BERT, можно скачать с https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
```
python evaluate.py [dataset] [task_type] [bert_checkpoint] [layer_count] [output_name]
```
Варианты аргумента task_type: sentiment, pos, wsi, anaphora.

Пример:
```
python evaluate.py datasets/sentiment sentiment ../cased_L-12_H-768_A-12/ 12 sentiment
```
Новый файл будет создан в директории `outputs`, там его будет искать main.py. 

Там уже есть несколько обработанных датасетов.