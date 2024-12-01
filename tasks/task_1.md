# Извлечение данных из коллекции новостных текстов

Данные расположены  [в файле data/news.tar.gz](data/news.tar.gz). С некоторых новостных сайтов был загружен архив новостей а  несколько лет, причем каждая новость принаделжит к какой-то рубрике: `science`, `style`, `culture`, `life`, `economics`, `business`, `travel`, `forces`, `media`, `sport`
    

В каждой строке файла содержится метка рубрики, заголовок новостной статьи и сам текст статьи, например:

        sport <tab> Сборная Канады по хоккею разгромила чехов <tab> Сборная Канады по хоккею крупно об...

С помощью [Yargy](https://github.com/natasha/yargy) или [Томита-парсера](https://github.com/yandex/tomita-parser) извлеките данные, которые можно описать структурой вида:


```python
@dataclass
class Entry:
    name: str
    birth_date: Optional[str]
    birth_place: Optional[str]


# 从新闻文本集中提取数据

数据位于 [data/news.tar.gz](data/news.tar.gz) 文件中。 从一些新闻网站下载了几年来的新闻档案，每条新闻都被分配到一个栏目： 科学"、"风格"、"文化"、"生活"、"经济"、"商业"、"旅游"、"部队"、"媒体"、"港口"。


文件的每一行都包含标题标签、新闻文章的标题和文章本身的文字，例如

        sport <tab> 加拿大曲棍球队击败捷克队 <tab> 加拿大曲棍球队大比分击败...

使用 [Yargy](https://github.com/natasha/yargy) 或 [Tomita 解析器](https://github.com/yandex/tomita-parser) 提取可以用以下结构描述的数据：


``python
数据类
类条目：
    name: str
    birth_date: 可选[str]
    birth_place: 可选[str]