# Dataset

The project was developed using a resume dataset containing resume text, job titles, and category labels.

The raw CSV is not included in this public repository because it may contain personal information and is also larger than GitHub's standard per-file upload limit.

Expected file name for local training:

`Resume dataset.csv`

Expected columns:
- `category`
- `job_title`
- `Text`

Place the CSV in this `data/` folder before running `train_model.py` locally.
