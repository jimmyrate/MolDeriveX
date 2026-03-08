import csv
from random import shuffle 
def read_csv_column(file_path, column_name,sample_num):
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # 检查列是否存在
            if column_name not in reader.fieldnames:
                print(f"列 '{column_name}' 不存在于CSV文件中.")
                return None

            # 读取指定列的数据
            column_data = [row[column_name] for row in reader]
            shuffle(column_data)
            return column_data[:sample_num]
    except FileNotFoundError:
        print(f"文件 '{file_path}' 不存在.")
        return None
    except csv.Error as e:
        print(f"读取CSV文件时发生错误: {e}")
        return None
    except Exception as e:
        print(f"发生未知错误: {e}")
        return None
    
#x_init = read_csv_column('/root/morbo/morbo/problems/postive_data_new_for_train.csv','SMILES',2847)
#print(len(x_init))