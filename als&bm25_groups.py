from utils import *

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from implicit.nearest_neighbours import bm25_weight
from implicit.als import AlternatingLeastSquares
from rank_bm25 import BM25Okapi

def main(args):
    # data
    group2id, id2group = read_groups_data()
    users_group_matrix, users, users_list = read_user_group_matrix(group2id)
    users_group_matrix = np.array(users_group_matrix)
    users_info = read_userinfo()

    # model
    ## als
    users_group_matrix = bm25_weight(users_group_matrix, K1=100, B=0.8)
    users_group_matrix = users_group_matrix.tocsr()
    model = AlternatingLeastSquares(factors=10, regularization=0.1, iterations=5000, calculate_training_loss=True, num_threads=0) # iterations=500
    model.fit(users_group_matrix)
    ## bm25
    corpus = []
    for i in users_list:
        corpus.append(users_info[i])
    tokenized_corpus = [doc.split(",") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    # test and output
    with open('data/test_unseen_group.csv', newline='') as csvfile:
        test_users_courses = csv.DictReader(csvfile)
        with open(f'{args.output_dir}/bm25_unseen_group.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["user_id", "subgroup"])
            for test_user in test_users_courses:
                query = users_info[test_user['user_id']]
                tokenized_query = query.split(",")
                doc_scores = bm25.get_scores(tokenized_query)
                simirality_user = users_list[np.argmax(doc_scores)]
                userid = users[simirality_user]
                ids, scores = model.recommend(userid, users_group_matrix[userid], N=90, filter_already_liked_items=False)
                writer.writerow([test_user['user_id'], ' '.join([str(id2group[id]) for id in ids])])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir",
        type=Path,
        help="Path to the output directory.",
        default="./outputs/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
