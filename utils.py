import csv
import numpy as np


def read_users_data():
    users = {}
    with open('data/users.csv', newline='') as csvfile:
        # 讀取 CSV 檔內容，將每一列轉成一個 dictionary
        users_info = csv.DictReader(csvfile)
        for user in users_info:
            users[user["user_id"]] = {
                "gender": user["gender"],
                "occupation_titles": user["occupation_titles"],
                "interests": user["interests"]
            }
    return users


def read_courses_data():
    course2id = {}
    id2course = {}
    with open('data/courses.csv', newline='') as csvfile:
        courses = csv.DictReader(csvfile)
        for i, course in enumerate(courses):
            course2id[course['course_id']] = i
            id2course[i] = course['course_id']
    return course2id, id2course


def read_groups_data():
    group2id = {}
    id2group = {}
    with open('data/subgroups.csv', newline='') as csvfile:
        group = csv.DictReader(csvfile)
        for i, subgroup in enumerate(group):
            group2id[subgroup['subgroup_id']] = i
            id2group[i] = subgroup['subgroup_id']
    return group2id, id2group


def read_train_courses_data(users, course2id):
    train_dataset = []
    train_users_course = {}     
    with open('data/train.csv', newline='') as csvfile:
        train = csv.DictReader(csvfile)
        for i in train:
            user_train = {}
            user_train["user_id"] = i["user_id"]
            user_train["interests"] = users[i["user_id"]]["interests"] + ',' +users[i["user_id"]]["occupation_titles"] if users[i["user_id"]]["occupation_titles"] != "" else users[i["user_id"]]["interests"]
            user_train["course_id"] = [course2id[course] for course in i["course_id"].split(" ")]
            train_users_course[i["user_id"]] = user_train["course_id"]
            train_dataset.append(user_train)
    return train_dataset#, train_users_course


def read_val_seen_courses_data(users, course2id):
    validation_dataset = []     
    with open('data/val_seen.csv', newline='') as csvfile:
        validation = csv.DictReader(csvfile)
        for i in validation:
            user_validation = {}
            user_validation["user_id"] = i["user_id"]
            user_validation["interests"] = users[i["user_id"]]["interests"] + ',' +users[i["user_id"]]["occupation_titles"] if users[i["user_id"]]["occupation_titles"] != "" else users[i["user_id"]]["interests"]
            user_validation["course_id"] = [course2id[course] for course in i["course_id"].split(" ")]
            validation_dataset.append(user_validation)
    return validation_dataset


def read_train_groups_data(users, group2id):
    train_dataset = []
    train_users_subgroup = {}   
    with open('data/train_group.csv', newline='') as csvfile:
        train = csv.DictReader(csvfile)
        for i in train:
            if i["subgroup"] == "":
                train_users_subgroup[i["user_id"]] = []
                continue
            user_train = {}
            user_train["user_id"] = i["user_id"]
            user_train["interests"] = users[i["user_id"]]["interests"] + ',' +users[i["user_id"]]["occupation_titles"] if users[i["user_id"]]["occupation_titles"] != "" else users[i["user_id"]]["interests"]
            user_train["subgroup"] = [group2id[subgroup] for subgroup in i["subgroup"].split(" ")]
            train_users_subgroup[i["user_id"]] = user_train["subgroup"]
            train_dataset.append(user_train)
    return train_dataset#, train_users_subgroup


def read_val_seen_groups_data(users, group2id):
    validation_dataset = []     
    with open('data/val_unseen_group.csv', newline='') as csvfile:
        validation = csv.DictReader(csvfile)
        for i in validation:
            if i["subgroup"] == "":
                continue
            user_validation = {}
            user_validation["user_id"] = i["user_id"]
            user_validation["interests"] = users[i["user_id"]]["interests"] + ',' +users[i["user_id"]]["occupation_titles"] if users[i["user_id"]]["occupation_titles"] != "" else users[i["user_id"]]["interests"]
            user_validation["subgroup"] = [group2id[subgroup] for subgroup in i["subgroup"].split(" ")]
            validation_dataset.append(user_validation)
    return validation_dataset


def read_test_courses_data(users, test_file_path):
    test_dataset = []     
    with open(test_file_path, newline='') as csvfile:
        test = csv.DictReader(csvfile)
        for i in test:
            user_validation = {}
            user_validation["user_id"] = i["user_id"]
            user_validation["interests"] = users[i["user_id"]]["interests"] + ',' +users[i["user_id"]]["occupation_titles"] if users[i["user_id"]]["occupation_titles"] != "" else users[i["user_id"]]["interests"]
            test_dataset.append(user_validation)
    return test_dataset


def write_test_courses_data(test_dataset, predicts_list, id2course, ouptut_file_path):
    with open(ouptut_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'course_id'])
        
        for i, predicts in enumerate(predicts_list):
            courses = " ".join([str(id2course[pred]) for pred in predicts])
            writer.writerow([test_dataset[i]["user_id"], courses])
            

def read_test_groups_data(users, test_file_path):
    test_dataset = []     
    with open(test_file_path, newline='') as csvfile:
        test = csv.DictReader(csvfile)
        for i in test:
            user_validation = {}
            user_validation["user_id"] = i["user_id"]
            user_validation["interests"] = users[i["user_id"]]["interests"] + ',' +users[i["user_id"]]["occupation_titles"] if users[i["user_id"]]["occupation_titles"] != "" else users[i["user_id"]]["interests"]
            test_dataset.append(user_validation)
    return test_dataset


def write_test_groups_data(test_dataset, predicts_list, id2group, ouptut_file_path):
    with open(ouptut_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['user_id', 'subgroup'])
        
        for i, predicts in enumerate(predicts_list):
            subgroup = " ".join([str(id2group[pred]) for pred in predicts])
            writer.writerow([test_dataset[i]["user_id"], subgroup])


def read_user_group_matrix(group2id):
    users_group_matrix = []
    users = {}
    users_list = []
    with open('data/train_group.csv', newline='') as csvfile:
        users_group = csv.DictReader(csvfile)
        for i, user in enumerate(users_group):
            group = np.zeros(91)
            if user['subgroup'] != '':
                for c in user['subgroup'].split(' '):
                    group[group2id[c]] = 1.0
            users[user['user_id']] = i
            users_list.append(user['user_id'])
            users_group_matrix.append(group)
    return users_group_matrix, users, users_list


def read_userinfo():
    users_info = {}
    with open('data/users.csv', newline='') as csvfile:
        u = csv.DictReader(csvfile)
        for user in u:
            users_info[user['user_id']] = user['interests']
    return users_info


def read_user_course_matrix(course2id):
    users_course_matrix = []
    users = {}
    users_list = []
    with open('data/train.csv', newline='') as csvfile:
        users_course = csv.DictReader(csvfile)
        for i, user in enumerate(users_course):
            course = np.zeros(728)
            if user['course_id'] != '':
                for c in user['course_id'].split(' '):
                    course[course2id[c]] = 1.0
            users[user['user_id']] = i
            users_list.append(user['user_id'])
            users_course_matrix.append(course)
    return users_course_matrix, users, users_list
