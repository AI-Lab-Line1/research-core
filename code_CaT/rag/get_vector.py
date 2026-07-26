from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community .embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def main():
    # 向量模型路径
    EMBEDDING_MODEL = './Jerry0--m3e-base/snapshots/master'

    # 加载文档
    loader = TextLoader('./物流信息.txt', encoding='utf-8')
    data = loader.load()

    # 切分文档
    text_split = RecursiveCharacterTextSplitter(chunk_size=128,chunk_overlap=4)
    split_data = text_split.split_documents(data)

    # 初始化huggingface模型embedding
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # 将切分的文档进行向量化并存储
    db = FAISS.from_documents(split_data,embeddings)
    db.save_local('./faiss/camp')

    return split_data

if __name__ == '__main__':
    split_data = main()
    print(f'split_data')
    for content in split_data:
        print(content.page_content)
        break