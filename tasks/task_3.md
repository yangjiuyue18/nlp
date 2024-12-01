# Применение инструментов Hugging face и предобученных моделей

## Вариант 1:
 Вам нужно создать искусственные данные для тестирования и/или обучения чат-бота. По заданному предложению/утверждению/команде создать набор расширенных предложений/утверждений/команд с приблизительно тем же смыслом. Пример:

> After your workout, remember to focus on maintaining a good water balance.

похожие команды:

> Remember to drink enough water to restore and maintain your body's hydration after your cardio training.

>Please don't forget to maintain water balance after workout.

Предлагается решить упрощенную версию данной задачи с применением общедоступных "маленьких". 
В репозитории Hugging Face есть большое количество предобученных моделей для [casual](https://huggingface.co/models?pipeline_tag=text-generation) и [masked](https://huggingface.co/models?pipeline_tag=fill-mask) языкового моделирования.  Также для валидации можно использовать [sentence-transformers](https://huggingface.co/sentence-transformers). Выбрать нужно модели, которые можно запускать на CPU.

Пример использования masked LM:

```python
import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer

# загружается токенайзер
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
# загружается модель
model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

# предложение и замаскированным токеном
sequence = f"My name is {tokenizer.mask_token}."

# результат токенизации
input_ids = tokenizer.encode(sequence, return_tensors="pt")
# применение модели
result = model(input_ids=input_ids)

# индекс замаскированного токена (NB может не совпадать с номером слова)
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# самый вероятный токен 
print(tokenizer.decode(result.logits[:, mask_token_index].argmax()))
```

или через [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)

```python
from transformers import pipeline

pipe = pipeline("fill-mask", model="distilroberta-base")

pipe("My name is <mask>.")
```

Casual LM через pipeline:

```python
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')

generator("Hello", max_length=10, num_return_sequences=5)
```

Один наивных способов решения задачи без дополнительного обучения - замаскировать, или вставить в исходную команду замаскированный токен, или обрезать часть команды и применить языковую модель. Результат можно валидировать с помощью [sentence-transformers](https://huggingface.co/sentence-transformers). 

## Вариант 2:

Нужно реализовать простейшую семантическую поисковую систему помощью векторного представления предложений/текстов.
1. Выбрать коллекцию текстовых документов (небольшое подмножество статей из Википедии (из дампа), новости, и т.п.).
2. Выбрать модель для получения векторных представлений (например [sentence-transformers](https://huggingface.co/sentence-transformers)).
3. Выбрать векторное хранилище (faiss, lancedb, qdrant, chroma, pgvector, redis и т.д.)
4. Реализовать поиск, (возможно с постфильтрацией) и продемонстрировать его работу. Индексация и поиск должны быть реализованы в виде отдельных скриптов с CLI.

Нельзя использовать LangChain. 


# 应用 Hugging Face 工具和预训练模型

## 任务选项 1：
您需要创建用于测试和/或训练聊天机器人的人工数据。根据给定的句子/声明/命令，生成一组具有大致相同含义的扩展句子/声明/命令。例如：

> After your workout, remember to focus on maintaining a good water balance.

相似的命令：

> Remember to drink enough water to restore and maintain your body's hydration after your cardio training.

> Please don't forget to maintain water balance after workout.

建议使用 Hugging Face 提供的公开预训练“小型”模型解决此简化版本的任务。Hugging Face 仓库中有大量适用于 [casual](https://huggingface.co/models?pipeline_tag=text-generation) 和 [masked](https://huggingface.co/models?pipeline_tag=fill-mask) 语言建模的预训练模型。此外，还可以使用 [sentence-transformers](https://huggingface.co/sentence-transformers) 进行验证。需要选择能够在 CPU 上运行的模型。

以下是使用 masked LM 的示例：

```python
import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
# 加载模型
model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

# 包含掩码标记的句子
sequence = f"My name is {tokenizer.mask_token}."

# 分词结果
input_ids = tokenizer.encode(sequence, return_tensors="pt")
# 应用模型
result = model(input_ids=input_ids)

# 掩码标记的索引 (注意：可能与单词编号不一致)
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# 最可能的标记
print(tokenizer.decode(result.logits[:, mask_token_index].argmax()))
```

或者通过 [pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)：

```python
from transformers import pipeline

pipe = pipeline("fill-mask", model="distilroberta-base")

pipe("My name is <mask>.")
```

Casual LM 示例（通过 pipeline 实现）：

```python
from transformers import pipeline, set_seed

generator = pipeline('text-generation', model='gpt2')

generator("Hello", max_length=10, num_return_sequences=5)
```

一个简单的解决方案是在初始命令中插入掩码标记，或截断部分命令并应用语言模型。生成的结果可以使用 [sentence-transformers](https://huggingface.co/sentence-transformers) 进行验证。

---

## 任务选项 2：

需要实现一个最简单的语义搜索系统，通过句子/文本的向量表示实现。

1. 选择一个文本文档集合（例如小规模的维基百科文章子集、新闻等）。
2. 选择一个用于获取向量表示的模型（例如 [sentence-transformers](https://huggingface.co/sentence-transformers)）。
3. 选择一个向量存储工具（例如 faiss、lancedb、qdrant、chroma、pgvector、redis 等）。
4. 实现搜索功能（可以包含后处理过滤），并演示其运行。索引和搜索功能需实现为独立的 CLI 脚本。

**禁止使用 LangChain**