from sklearn.model_selection import train_test_split
import pickle

INPUT_DATA = "train.txt"
SAVE_PATH = "./datasave.pkl"
id2tag = ['B', 'M', 'E', 'S']  # B：分词头部 M：分词词中 E：分词词尾 S：独立成词
tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
word2id = {}
id2word = []  # 字的序列（字按出现顺序排列，位置即id）


def getList(input_str):
    """
    单个分词转换为tag序列
    :param input_str: 单个分词
    :return: tag序列
    """
    output_str = []
    if len(input_str) == 1:
        output_str.append(tag2id['S'])
    elif len(input_str) == 2:
        output_str = [tag2id['B'], tag2id['E']]
    else:
        M_num = len(input_str) - 2
        M_list = [tag2id['M']] * M_num
        output_str.append(tag2id['B'])
        output_str.extend(M_list)
        output_str.append(tag2id['E'])
    return output_str


def handle_data():
    """
    处理数据，并保存至savepath
    数据形式：
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 7, 22, 23, 24, 25, 18, 10, 11, 12, 13
    , 26, 27, 28, 7, 29, 30, 31, 28, 18, 32, 33, 18, 34, 35, 36, 37]
    [3, 0, 2, 3, 3, 0, 2, 3, 3, 3, 0, 1, 2, 3, 3, 3, 3, 3, 3, 0, 2, 3, 3, 0, 1, 1, 2, 3, 0, 1, 2, 3, 3, 0, 2, 3, 0, 1, 1
    , 2, 3, 3, 3, 3, 0, 2, 3, 3]
    :return:
    """
    x_data = []
    y_data = []
    wordnum = 0
    line_num = 0
    with open(INPUT_DATA, 'r', encoding="utf-8") as ifp:
        for line in ifp:
            line_num = line_num + 1
            line = line.strip()
            if not line:
                continue
            line_x = []  # 字在字序列中的序号
            for i in range(len(line)):
                if line[i] == " ":
                    continue
                if line[i] in id2word:
                    line_x.append(word2id[line[i]])
                else:
                    id2word.append(line[i])
                    word2id[line[i]] = wordnum
                    line_x.append(wordnum)
                    wordnum = wordnum + 1
            x_data.append(line_x)

            lineArr = line.split()
            line_y = []
            for item in lineArr:
                line_y.extend(getList(item))
            y_data.append(line_y)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=43)
    with open(SAVE_PATH, 'wb') as outs:
        pickle.dump(word2id, outs)
        pickle.dump(id2word, outs)
        pickle.dump(tag2id, outs)
        pickle.dump(id2tag, outs)
        pickle.dump(x_train, outs)
        pickle.dump(y_train, outs)
        pickle.dump(x_test, outs)
        pickle.dump(y_test, outs)


if __name__ == "__main__":
    # 执行时间30多s
    handle_data()
