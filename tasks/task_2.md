
# Контекстно-независимое векторное представление слов 

## Вариант 1

1. Используйте тексты из предыдущего задания и обучите на них модель векторного представления слов (опционально можно приводить слова к нормальной форме и удалить стоп-слова). Можно использовать `gensim`.

2. Разделите коллекцию текстов на обучающее и тестовое множество. С помощью обученной модели векторного представления отобразите каждый документ в вектор, усреднив все вектора для слов документа. 

3. Используйте какой-либо алгоритм классификации (например `SVM`) для классификации текстов. Для обучения используйте тестовое множество, для анализа результатов - тестовое.

4. Простое усреднение векторов слов обычно не дает хорошего отображения документа в вектор. Придумайте альтернативный способ. Протестируйте его, повторно обучив алгоритм классификации на тех же данных. 

## Вариант 2

Нужно создать некоторый аналог утилиты [Grep](https://ru.wikipedia.org/wiki/Grep) для поиска строк в текстовом файле, который бы учитывал похожие по смыслу слова.
Grep применяется для поиска строк в текстовым файле, которые соответствуют/содержат какой-то паттерн.  Упрощенно - содержат какое-то слово.

Например команда
	> grep “привет” data.txt
 или
    > cat data.txt | grep “привет”     
выведет все строки файла data.txt, которые содержат “привет”.

Требуется реализовать утилиту на языке Python,  но в отличие от стандартного grep, утилита должна поддерживать поиск не только по точным совпадениям, но и с учетом  синонимов, для чего можно использовать модель Word2Vec или аналогичные неконтекстные векторные представления слов.

Например команда:
> python3 mygrep.py data.txt “привет”

В идеальном варианте должна вывести все строки в файле data.txt, которые содержат слова “привет”, “здравствуйте” и т.п.

Для обучения и применения Word2Vec можно использовать библиотеку gensim и набор текстов из предыдущего задания. Подумайте, как обрабатывать случаи, когда синонимы слова не найдены или если модель Word2Vec не содержит (OOV- out of vocabulary) введенное слово. В коде нужно предусмотреть дополнительную обработку ошибок, связанных с чтением файла, загрузкой модели и т.п.



# 基于上下文无关的词向量表示

## 任务选项 1

1. 使用上一个任务中的文本数据，训练一个词向量表示模型（可以选择将词归一化为基本形式，并移除停用词）。可以使用 `gensim` 库。

2. 将文本集合划分为训练集和测试集。使用训练好的词向量模型，将每篇文档映射为一个向量，方法是对文档中所有词的向量取平均。

3. 使用某种分类算法（例如 `SVM`）对文本进行分类。用训练集进行模型训练，用测试集分析结果。

4. 简单的词向量平均通常不能很好地表示文档。设计一个替代方法，重新在相同数据上训练分类算法并测试。

## 任务选项 2

需要创建一个类似于 [Grep](https://ru.wikipedia.org/wiki/Grep) 的工具，用于在文本文件中搜索字符串，但要支持语义相似词的匹配。  
Grep 是一个用于在文本文件中搜索包含某种模式（或某个单词）的行的工具。  

例如，命令：
```bash
> grep “привет” data.txt
```
或
```bash
> cat data.txt | grep “привет”
```
将输出文件 `data.txt` 中包含“привет”的所有行。

需要使用 Python 实现一个工具，但与标准 `grep` 不同，该工具应该支持根据词义相近的单词进行搜索。为此可以使用 Word2Vec 模型或类似的非上下文相关的词向量表示。

例如，命令：
```bash
> python3 mygrep.py data.txt “привет”
```
理想情况下应该输出文件 `data.txt` 中包含“привет”、“здравствуйте”等词的所有行。

### 任务要求：
1. 使用 `gensim` 库和上一个任务中的文本数据来训练和应用 Word2Vec 模型。
2. 思考如何处理以下情况：
   - 输入单词的同义词未找到。
   - Word2Vec 模型中不包含输入单词（OOV - out of vocabulary）。
3. 代码中需要提供错误处理机制，例如：
   - 文件读取错误。
   - 模型加载错误。