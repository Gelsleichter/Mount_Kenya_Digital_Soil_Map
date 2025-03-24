# Mount Kenya Digital Soil Map 🌍

![Mount Kenya](https://via.placeholder.com/800x200.png?text=Mount+Kenya+Digital+Soil+Map)  
*Digital Soil Mapping for Mount Kenya Region*

Welcome to the **Mount Kenya Digital Soil Map** repository! This project focuses on creating a digital soil map for the Mount Kenya region using machine learning techniques to predict Soil Organic Carbon (SOC) levels. The repository contains R scripts for data processing, model training, validation, and visualization of SOC distributions across different datasets.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data Description](#data-description)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🌟 Overview

This project aims to develop a digital soil map for the Mount Kenya region by predicting Soil Organic Carbon (SOC) levels using machine learning. The workflow includes:
- **Data Preprocessing:** Cleaning and preparing soil data for analysis.
- **Model Training:** Using cross-validation to train models on training and validation datasets.
- **Evaluation:** Assessing model performance on test datasets.
- **Visualization:** Generating density plots and scatterplots to analyze SOC distributions and model predictions.

The scripts in this repository are written in R and utilize packages like `ggplot2` and `tune` for visualization and model evaluation.

---

## 🛠️ Installation

To run the scripts in this repository, you'll need to have R installed on your system. Follow these steps to set up the environment:

1. **Install R:**
   - Download and install R from [CRAN](https://cran.r-project.org/).

2. **Install RStudio (Optional):**
   - For a better development experience, install [RStudio](https://www.rstudio.com/products/rstudio/download/).

3. **Run the GEE script to get the covariates:**
   Open R or RStudio and run the scrips from the repo:

4. **Install Required R Packages:**
   Open R, preferably RStudio and run the scrips from the repo:
   
## 📊 Data Description

The project uses four datasets to predict Soil Organic Carbon (SOC) levels in the Mount Kenya region:

- **Train-CV:** Combined training and validation data used in cross-validation (50 samples).
- **Validation:** Data for hyperparameter tuning (10 samples).
- **Test:** Data for final model evaluation (10 samples).

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or feedback, feel free to reach out:

- **GitHub:** [Gelsleichter](https://github.com/Gelsleichter)
- **Email:** [gelsleichter.yuri.andrei(at)uni-mate.hu](mailto:gelsleichter.yuri.andrei@uni-mate.hu)
