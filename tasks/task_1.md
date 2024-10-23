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