# Recommendation
use HuggingFace BERT-Chinese model to recommend courses or subgroups to users

## Packages installation
Install python packages
```python
pip install -r requirements.txt
```
## Data
### users data
- user_id, gender, occupation_titles, interests, recreation_names
### users purchase courses data
- user_id, course_id
### courses data
- course_id, course_name, course_price, teacher_id, teacher_intro, groups, sub_groups, topics

## Models
### multi-label BERT
- Courses
```python
python bert_courses.py
```
- Subgroups
```python
python bert_groups.py
```

### ALS + BM25 recommendation
- Courses
```python
python als_bm25_courses.py
```
- Subgroups
```python
python als_bm25_groups.py
```
