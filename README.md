# Oracle: Datathon 2020
---
>Our team's files for "Deep Learning Datathon 2020", organised by ai4impact.
> Team _Oracle_

Suggested way to use and edit `./src/*.py` files and notebook templates in `./nbs`:
- Copy notebook templates to a new folder, then create symbolic link e.g.:
    - `ln -s '../datathon2020/src/' .`
- Once you created the link, in your notebooks you can import all the modules in `./src` with `from src.[MODULE_NAME] import [SOMETHING]` e.g.:
    - `from src.datautils import windowed_data`
- If you develop a new and useful routine (e.g. implement prediction notebook, or add model saving, early stopping) edit the template notebook in this repository. Don't forget to share with others :smile:, and tell about the new routines that you add :loudspeaker:.

---
- `./src` python source code (2nd part of datathon).
- `./nbs` notebook templates for normalising, preprocessing, training and prediction steps.
- `./lab1` and `./lab2` are self-contained codes and notebooks for labs 1 and 2 (1st part of the datathon)
- Datasets:
    - _lab 1_
        - Raw training and test datasets in `./lab1/sg_temps` (`*.csv` files)
        - Raw `sg-temps` dataset stats: `lab1/sg_temps_stats.csv`
    - _lab2_  datasets and stats:
        - Raw dataset (has missing values for some hours): `./lab2/data/PJM_Load_hourly.csv`
        - Raw dataset stats: `./lab2/data_stats.csv`
        - Test and training datasets (not normalized data): `./lab2/data/test.csv` and `./lab2/data/train.csv`
