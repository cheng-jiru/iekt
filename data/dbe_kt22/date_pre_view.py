import pickle
import json
with open('history_test.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
# print(loaded_data)
# 2. 将数据保存为json文件
with open('history_test.json', 'w', encoding='utf-8') as file:
    json.dump(loaded_data, file, ensure_ascii=False, indent=4)

print("数据已成功从pkl转换为json格式并保存")