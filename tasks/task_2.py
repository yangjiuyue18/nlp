import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

def train_or_load_model(data_path, model_path="word2vec.model"):
    """
    训练或加载 Word2Vec 模型。
    
    参数:
        data_path (str): 文本文件路径，用于训练模型。
        model_path (str): Word2Vec 模型的保存路径。
    
    返回:
        gensim.models.Word2Vec: 训练好的 Word2Vec 模型。
    """
    if os.path.exists(model_path):
        print(f"加载已存在的模型: {model_path}")
        try:
            return Word2Vec.load(model_path)
        except Exception as e:
            print(f"[ERROR] 无法加载模型: {e}")
            exit(1)
    
    # 读取文件内容
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 分词
    sentences = [word_tokenize(line.lower()) for line in lines]

    # 训练 Word2Vec 模型
    print("训练 Word2Vec 模型...")
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
    model.save(model_path)
    print(f"模型已保存: {model_path}")
    return model

def find_similar_words(word, model, topn=10):
    """
    查找与目标单词语义相近的单词。

    参数:
        word (str): 输入单词。
        model (gensim.models.Word2Vec): Word2Vec 模型。
        topn (int): 返回的相似单词数量。
    
    返回:
        list: 相似单词列表（包括原始单词）。
    """
    if word not in model.wv:
        print(f"[INFO] 单词 '{word}' 不在词汇表中 (OOV)，仅搜索原始单词。")
        return [word]
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        return [word] + [sim[0] for sim in similar_words]
    except Exception as e:
        print(f"[ERROR] 查找相似单词时出错: {e}")
        return [word]

def search_in_file(data_path, words):
    """
    在文件中搜索包含指定单词的行。

    参数:
        data_path (str): 文本文件路径。
        words (list): 要搜索的单词列表。
    
    返回:
        list: 匹配的行。
    """
    matches = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if any(word in line.lower() for word in words):
                matches.append(line.strip())
    return matches

def document_to_vector(doc, model):
    """
    将文档转换为向量表示（词向量平均）。

    参数:
        doc (str): 文档文本。
        model (gensim.models.Word2Vec): 词向量模型。
    
    返回:
        np.array: 文档的向量表示。
    """
    words = word_tokenize(doc.lower())
    vectors = [model.wv[word] for word in words if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)  # 如果文档没有已知单词，返回零向量
    return np.mean(vectors, axis=0)

def classify_texts(data_path, model):
    """
    使用 SVM 对文本进行分类。

    参数:
        data_path (str): 文本数据路径。
        model (gensim.models.Word2Vec): 词向量模型。
    """
    # 读取数据
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # 假设每行的格式为 "标签\t文本"
    labels, texts = zip(*(line.split('\t', 1) for line in lines if '\t' in line))
    
    # 转换文本为向量
    vectors = np.array([document_to_vector(text, model) for text in texts])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(vectors, labels, test_size=0.2, random_state=42)

    # 训练 SVM 分类器
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)

    # 测试分类器
    y_pred = classifier.predict(X_test)
    print(classification_report(y_test, y_pred))

def main():
    data_path = '../data/news.txt'
    model_path = 'word2vec.model'
    target_word = input("请输入搜索的目标单词: ").strip().lower()

    # 加载或训练模型
    model = train_or_load_model(data_path, model_path)

    # 语义搜索
    similar_words = find_similar_words(target_word, model)
    print(f"与 '{target_word}' 语义相近的单词: {', '.join(similar_words)}")

    matches = search_in_file(data_path, similar_words)
    print("\n匹配的行:")
    if matches:
        for match in matches:
            print(match)
    else:
        print("未找到匹配的行.")

    # 分类
    print("\n执行文本分类任务...")
    classify_texts(data_path, model)

if __name__ == "__main__":
    main()
