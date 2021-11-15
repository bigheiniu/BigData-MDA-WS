import pandas as pd
import json
def read_file(file_path, save_path):
    data_list = []
    for idx, type in enumerate(['real','fake']):
        data = pd.read_csv(file_path.format(type))
        data['label'] = idx
        data_list.append(data)

    data_df = pd.concat(data_list, axis=0)
    data_df.to_csv(save_path, index=None)

def read_json(file_path, label):
    data = json.load(open(file_path,'r'))
    content = [i['text_content'] for i in data['dataset']]
    comment = ["|".join([j['text'] for j in i['tweets']]) for i in data['dataset']]
    data_df = [(i.replace("\n",""),j.replace("\n",""),label) for i, j in zip(content, comment) if len(comment) > 10 and len(content) > 0]
    data_df = pd.DataFrame(data_df, columns=["text","comment","label"])
    return data_df
if __name__ == '__main__':
    # file_path = "/home/yli29/fake_news_crawler/DataAnalysis/{}_news_source.txt"
    # save_path = "../data/covid_19_content.csv"
    # read_file(file_path, save_path)
    file_path = "/home/dmahudes/Fake_News_Detection/data/politifact_{}_news_dataset.json"
    data_df = []
    for index, type in enumerate(['real','fake']):
        data_df.append(read_json(file_path.format(type), index))
    data_df = pd.concat(data_df)
    data_df.to_csv("../data/politi_text_comment.csv", index=None)
