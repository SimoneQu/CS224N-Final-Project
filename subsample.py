import json
import random
import copy
import os


def subsample(data_file, subsample_num):
    squad_data = json.load(open(data_file, 'rb'))
    # Count number of questions
    all_ids = []
    for example in squad_data["data"]:
        for paragraph in example["paragraphs"]:
            for question_answer in paragraph["qas"]:
                all_ids.append(question_answer['id'])

    chosen_ids = set(random.sample(all_ids, k=subsample_num))
    subsampled_data = copy.deepcopy(squad_data)
    # Verify the data
    for article in subsampled_data["data"]:
        for paragraph in article["paragraphs"]:
            new_qas = []
            for qas in paragraph["qas"]:
                if qas['id'] in chosen_ids:
                    new_qas.append(qas)
            paragraph['qas'] = new_qas

    with open(data_file + '_subsampled', "w") as output_file:
        json.dump(subsampled_data, output_file)

def subsample_by_example(folder, file, pct, folder_to_write, file_to_write=None):
    data_file = os.path.join(folder, file)
    squad_data = json.load(open(data_file, 'rb'))
    print(f"Reading in {file}")
    n = len(squad_data['data'])
    print("Total count of data:", n)

    # Count number of questions
    subsample_num = int(n * pct) + 1
    subsample = random.sample(squad_data['data'], subsample_num)
    squad_data['data'] = subsample

    if file_to_write is None:
        file_to_write = file
    dir_to_write = os.path.join(folder_to_write, file_to_write)
    print(f"writing {subsample_num} subsamples to {dir_to_write}")
    with open(dir_to_write, "w") as output_file:
        json.dump(squad_data, output_file)

def count(data_file):
    squad_data = json.load(open(data_file, 'rb'))
    # Count number of questions
    all_ids = []
    count1 = 0
    for example in squad_data["data"]:
        flag = 0
        for paragraph in example["paragraphs"]:
            for question_answer in paragraph["qas"]:
                flag = 1
                all_ids.append(question_answer['id'])
        count1 += flag

    print(data_file)
    print("total number of examples:", len(squad_data["data"]))
    print("total number of examples with questions:", count1)
    print("total number of questions:", len(all_ids))


if __name__ == '__main__':
    random.seed(2021)
    folder = os.path.join(os.getcwd(), "datasets", "indomain_train")
    # folder = os.path.join(os.getcwd(), "datasets", "indomain_val")
    folder_to_write = os.path.join(os.getcwd(), "datasets_subsample", "indomain_train")
    if not os.path.exists(folder_to_write):
        os.mkdir(folder_to_write)
    for filename in ['nat_questions', 'newsqa', 'squad']:
        # dir = os.path.join(folder, filename)
        # count(dir)
        # subsample(dir, 500)
        subsample_by_example(folder, filename, 0.01, folder_to_write)
