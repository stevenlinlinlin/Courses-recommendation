# Recommendation
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
Course/Subgroup recommendation in BERT repository
- Courses
```python
python bert_courses.py
```
- Subgroups
```python
python bert_groups.py
```
### BM25
Course/Subgroup recommendation in BM25 repository
```python
python bm25_subgroup.py
```

### XGBoost
Course recommendation in XGBoost repository
```python
python xgb_cls.py
```
