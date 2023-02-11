# Courses recommendation
use machine learning models to recommend courses or subgroups to users

## Data
### users data
- user_id, gender, occupation_titles, interests, recreation_names
### users purchase courses data
- user_id, course_id
### courses data
- course_id, course_name, course_price, teacher_id, teacher_intro, groups, sub_groups, topics

## Models
### multi-label BERT
Course/Subgroup recommendation
```python
python bert_cls.py
```
### BM25
Subgroup recommendation
```python
python bm25_subgroup.py
```

### XGBoost
Course recommendation
```python
python xgb_cls.py
```
