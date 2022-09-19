# Технологии нейросетевых вычислений

## Лабораторные работы

1. [Детектор тунцов](lab1-tuna)

## Настройка рабочей среды

Установить необходимую версию Pytorch: https://pytorch.org/

Установить остальные зависимости:

```shell
pip install -r requirements.txt
```

Добавить git hook:
```shell
pre-commit install
```

Скачать датасет с [fishnet.ai](https://www.fishnet.ai/download). Положить images и labels в соответствующие директории в `lab1-tuna/resources`.
