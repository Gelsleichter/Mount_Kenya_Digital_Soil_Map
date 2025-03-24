# Mount Kenya Digital Soil Map 🌍

![Mount Kenya](https://upload.wikimedia.org/wikipedia/commons/b/b3/Terek_Valley_Mt_Kenya.jpg)
*Digital Soil Mapping for Mount Kenya Region*

Welcome to the **Mount Kenya Digital Soil Map** repository! This [repository](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/) presents the scripts to creatie a digital soil map for the Mount Kenya region using machine learning techniques to predict Soil Organic Carbon (SOC) levels. The [repository](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/) contains R scripts for data processing, model training, validation, and visualization of SOC and map creation.

---

## 📋 Table of Contents
- Overview
- Scripts
- Data Description
- Contact

---

## 🌟 Overview

This [repository](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/) contains the scripts to develop a digital soil map for the Mount Kenya region to predict the Soil Organic Carbon (SOC) levels using machine learning. The workflow includes:
- **Data Preprocessing:** Cleaning and preparing soil data for analysis.
- **Model Training:** Using cross-validation to train models on training and validation datasets.
- **Evaluation:** Assessing model performance on test datasets.
- **Uncertainty:** Assessing Uncertainty with quantile forest with ranger package.
- **Visualization:** Generating density plots, scatterplots and maps with the SOC spatial distributions.

The scripts in this [repository](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/) are written in R and JavaScript for [Google Earth Engine (GEE)](https://earthengine.google.com/).

---

## 🛠️ Scripts

To run the scripts in this [repository](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/), you'll need to have R installed on your system. Follow these steps to set up the environment:

1. **Install R:**
   - Download and install R from [CRAN](https://cran.r-project.org/).

2. **Install RStudio (Optional):**
   - For a better development experience, install [RStudio](https://www.rstudio.com/products/rstudio/download/).

3. **Run the GEE script to get the covariates:**
   Create an account in [Google Earth Engine (GEE)](https://earthengine.google.com/) and run the following scrip:
   [Mount_Kenya_covariates_preparation](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/blob/main/Mount_Kenya_covariates_preparation_v5.js)

5. **Install Required R Packages:**
   Open R, preferably RStudio and run the scrips from the repo:
   [Extract_points](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/blob/main/Extract_points.R)
   [Modeling_quantile_predict_maps](https://github.com/Gelsleichter/Mount_Kenya_Digital_Soil_Map/blob/main/Modeling_quantile_predict_maps.R)
   
---

## 📊 Data Description

To predict Soil Organic Carbon (SOC) in the Mount Kenya region the dataset was split into:

- **Train-CV:** Combined training and validation data used in cross-validation (50 samples).
- **Validation:** Data for hyperparameter tuning (10 samples).
- **Test:** Data for final model evaluation (10 samples).

---

## 📬 Contact

For questions or feedback, feel free to reach out:

- **GitHub:** [Gelsleichter](https://github.com/Gelsleichter)
- **Email:** [gelsleichter.yuri.andrei(at)uni-mate.hu](mailto:gelsleichter.yuri.andrei@uni-mate.hu)

---

## 📜 License

Free to use and distribute.
