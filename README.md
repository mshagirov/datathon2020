# Oracle: Datathon 2020
---
>Our team's files for "Deep Learning Datathon 2020", organised by ai4impact.

**Team Name**: Oracle

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
    - _NN102_ (week 2)
      - `./NN102/sg_temps` containes raw (`sg_temps_raw.csv`) and normalised (`sg_temps.csv`), data was normalised by shifting it by mean $\mu\approx28.0$ and scaling it by the S.D. $\sigma\approx0.8$. $$x_{norm}=\frac{x_{raw}-\mu}{\sigma}$$.
    - _P003_ (dataset for the challenge)
      - `./P003/datasets/` contains:
        - Energy Generation Data for Ile-de-France, raw files from [RET](https://www.rte-france.com/) (real-time data will be updated when you re-download it in `./P003/introduction.ipynb`):
          - Units for power are in `MW`
          1. `eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2017.xls`
          1. `eCO2mix_RTE_Ile-de-France_Annuel-Definitif_2018.xls`
          1. `eCO2mix_RTE_Ile-de-France_En-cours-Consolide_FIXED_ERRCOLS.xls` (this is clean version of `eCO2mix_RTE_Ile-de-France_En-cours-Consolide.xls` data after removing empty columns)
          1. `eCO2mix_RTE_Ile-de-France_En-cours-TR.xls` (near real-time data, date and time are in Paris time)
        - Wind Forecast Data (from [Terra Weather](http://www.terra-weather.com/)) from two weather models for major wind farm locations in Ile-de-France region:
          1. `/model1/` contains latest forecasts for model 1.
          1. `/model2/` latest forecasts for model 2.
          1. `/historical1/` historical forecasts for model 1.
          1. `/historical2/` historical forecasts for model 2.
