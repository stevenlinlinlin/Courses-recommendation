# Recommendation
use HuggingFace(BERT-Chinese) model, [BM25](https://github.com/dorianbrown/rank_bm25), [Alternating Least Square](https://github.com/benfred/implicit) to recommend courses or subgroups to users

## Packages installation
Install python packages
```python
pip install -r requirements.txt
```
## Data
the following data(csv) is in data folder.
### Train data
#### users data
- user_id, gender, occupation_titles, interests, recreation_names
#### users purchase courses data
- user_id, course_id
#### courses data
- course_id, course_name, course_price, teacher_id, teacher_intro, groups, sub_groups, topics

### Test data
- seen users_id data
- unseen users_id data

## Models
### Multi-label BERT
I use HuggingFace(BERT-Chinese) model to train multi-label classification model to predict courses or subgroups for users.
- Courses
```python
python bert_courses.py --output_dir [output_dir]
```
- Subgroups
```python
python bert_groups.py --output_dir [output_dir]
```

### ALS + BM25 recommendation
For unseen data, I use BM25 to find smiliar seen user_id and recommend  courses or subgroups to unseen user_id with ALS.
- Courses
```python
python als_bm25_courses.py --output_dir [output_dir]
```
- Subgroups
```python
python als_bm25_groups.py --output_dir [output_dir]
```
