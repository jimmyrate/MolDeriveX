import csv 

path = "/root/morbo/morbo/problems/postive_data_new_for_train.csv"
mol = []
with open(path) as file:
    csv_reader = csv.reader(file)

    for row in csv_reader:
        mol.append(row[0])
mol = mol[1:]
print(mol[0])