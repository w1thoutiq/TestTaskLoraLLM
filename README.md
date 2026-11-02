# Перенос стиля Telegram-каналов (Qwen2.5-3B + QLoRA)

Метрики, результаты и выводы — в [RESULTS.md](RESULTS.md).

## Структура

```
config.yaml             все гиперпараметры и константы
RESULTS.md              отчёт по ТЗ
inference.sh            инференс: vLLM + LoRA

src/
  config.py             config.yaml → Settings (pydantic)
  data.py               train / val / test одной таблицей
  prompts.py            диалог для модели
  model.py              токенизатор и 4-bit Qwen
  generation.py         генерация с адаптером и без
  metrics.py            метрики
  train.py              обучение → models/lora-adapter/
  test.py               сравнение 4 вариантов → artifacts/

data/                   данные из ТЗ
artifacts/              outputs_type1.txt, outputs_type2.txt, train_log.csv, generations_all.json
```

## Запуск

```bash
pip install -r requirements.txt
python src/train.py
python src/test.py
```

## Инференс

```bash
./inference.sh
```
```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"style","temperature":0.05,"messages":[{"role":"system","content":"Ты редактор новостного Telegram-канала. Переписывай новость в стиле канала. Не меняй и не выдумывай цифры, даты, имена и факты."},{"role":"user","content":"Перепиши новостной текст в стиле Telegram-канала @banki_oil.\n\nЦентральный банк Российской Федерации совместно с тринадцатью коммерческими банками перешел ко второму этапу масштабного тестирования национальной платформы цифрового рубля."}]}'
```
