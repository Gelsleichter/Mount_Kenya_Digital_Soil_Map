#############################################################################################
### Mapping Mount Kenya Region 
### Yuri Andrei Gelsleichter
### May 2025 
#############################################################################################

### Set working directory automatically 
library(this.path)
path_ <- this.dir()
setwd(path_)

### Check the "new" working directory 
getwd()

### clean memory
# gc(); rm(list=ls())

### Set options
options(scipen = 999, digits = 7)

library(terra) # for rasters and vectors
library(dplyr) # for distinct function
library(caret)
library(ranger)
library(parallel)
library(doParallel)
# install.packages("fst")
library(fst)
library(tidyr) # drop NA
library(gridExtra) # for grid.arrange

dir("../../")
dir("../../0_Data_source_metadata_data_treatement")
dir("../../1_Covariates_preparation/output/")
dir("../../2_Extracting_points/output/")

### Load the data
### With "fst" faster and lighter
system.time({ # .5 second to read
df <- read_fst("../../2_Extracting_points/output/dataset_mod_22_covariates_df.fst")
})

table(df$data_type)

df <- df |> dplyr::filter(data_type == "training")

colSums(is.na(df)) |> sort()
df <- subset(df, select = -c(landcover))

# Drop NA
df <- df |> tidyr::drop_na()

################################################################################
### Covariates from rfe 
################################################################################
opt_vars_r1f <- readLines("../output/rfe/round1/covariates_selection_rfe_RF_rerank_false.txt")

################################################################################
### Adjust the train, test data
# Add data_type column
# df <- add_column(df, data_type = NA, .after = 1)

# Create a vector with elements "train" and "test"
# 70*.85; 70*.15 # approx 85 and 15% of the data
# dt <- c(rep("train", 60), rep("test", 10)) # approx 85 and 15% of the data
dt <- c(rep("train", 50), rep("valid", 10), rep("test", 10)) # approx 85 and 15% of the data
# Randomly shuffle the vector
set.seed(123)
random_dt <- sample(dt)

# Assign the shuffled vector to the data_type column
df$data_type <- random_dt

# Compose and select columsn
df.mod <- df[ , c("SOC", "data_type", opt_vars_r1f)]
dim(df.mod)

df.mod <- subset(df.mod, select = -c(landform, x, y))

df_mod_train <- df.mod |> dplyr::filter(data_type == "train")
df_mod_valid <- df.mod |> dplyr::filter(data_type == "valid")
df_mod_test <- df.mod |> dplyr::filter(data_type == "test")

train_predictors <- df_mod_train |> dplyr::select(!c(SOC, data_type))
valid_predictors <- df_mod_valid |> dplyr::select(!c(SOC, data_type))
test_predictors <- df_mod_test |> dplyr::select(!c(SOC, data_type))

train_response <- df_mod_train |> dplyr::select(c(SOC)) |> pull(SOC) # pull(SOC) is teh same as [, drop=T]
valid_response <- df_mod_valid |> dplyr::select(c(SOC)) |> pull(SOC)
test_response <- df_mod_test |> dplyr::select(c(SOC)) |> pull(SOC)

################################################################################
### Filter data train test
################################################################################
table(df$data_type)

# Combine train and validation data for 10-fold cross-validation
train_v_predictors <- rbind(train_predictors, valid_predictors)
train_v_response <- c(train_response, valid_response)

### train_v_predictors <- rbind(train_predictors, valid_predictors, test_predictors)
### train_v_response <- c(train_response, valid_response, test_response)
dim(train_v_predictors)
table(is.na(train_v_predictors))

################################################################################
# Setings for parallel processing
n_cores <- detectCores() # - 1 # leave 1 core for OS
registerDoParallel(cores = n_cores) # Use mclapply internally in Linux

{
######## Config ########
# Number of folds and repetitions
k <- 15 # https://doi.org/10.1016/j.geodrs.2024.e00901
repeats <- 100
n <- nrow(train_v_predictors) # Number of observations
total_iterations <- k * repeats # Total number of models (1000)
# Counter for resample naming
resample_counter <- 1 # start at 1

# Define parameters
resample_counter <- 1 # Reset counter
num_trees <- 1000
mtry <- 12
min_node_size <- 10
sampe_size <- 0.9
max_depth <- 0
splitrule <- "variance"
importance <- "permutation"

# Pre-allocate lists (usualy most efficient)
# models_list <- vector("list", total_iterations) # if store ranger models (2GB)
row_index_ls <- vector("list", total_iterations)
Fold_ls <- vector("list", total_iterations)
Resample_ls <- vector("list", total_iterations)
rsq_cal_ls <- vector("list", total_iterations)
mse_cal_ls <- vector("list", total_iterations)
rmse_cal_ls <- vector("list", total_iterations)
mae_cal_ls <- vector("list", total_iterations)
rsq_val_ls <- vector("list", total_iterations)
mse_val_ls <- vector("list", total_iterations)
rmse_val_ls <- vector("list", total_iterations)
mae_val_ls <- vector("list", total_iterations)

system.time({
# Loop for 100 repetitions
for (r in 1:repeats) {
  # Create indices for k-fold CV
  folds <- createFolds(train_v_response, k = k, list = TRUE, returnTrain = FALSE)
  
  # Loop for each fold
  for (fold_idx in 1:k) {
    # Resample name (e.g., Resample0001, Resample0002, ...)
    resample_name <- sprintf("Resample%04d", resample_counter)
    i <- resample_counter # call it 'i' for simplicity
    
    # Test and train indices
    valid_idx <- folds[[fold_idx]]
    train_idx <- setdiff(1:n, valid_idx)
    
    # Split data into training and test sets
    train_predictors_int <- train_v_predictors[train_idx, ]
    train_response_int <- train_v_response[train_idx]
    valid_predictors_int <- train_v_predictors[valid_idx, ]
    valid_response_int <- train_v_response[valid_idx]
    
    # Train the Quantile Random Forest model with ranger
    qrf_model <- ranger::ranger(
      x                 = train_predictors_int,
      y                 = train_response_int,
      num.trees         = num_trees,
      mtry              = mtry,
      min.node.size     = min_node_size,
      sample.fraction   = sampe_size,
      max.depth         = max_depth,
      splitrule         = splitrule,
      # num.random.splits = num_random_splits, # for extratrees only
      importance        = importance,
      quantreg          = TRUE, # Enable Quantile Regression Forest
      replace           = TRUE,
      num.threads       = n_cores 
    )
    
    # Store model
    # models_list[[i]] <- qrf_model
    
    # Predict median (quantile 0.5)
    # Calibration predictions (on training set)
    pred_cal <- predict(qrf_model, data = train_predictors_int, 
                        type = "quantiles", quantiles = 0.5)$predictions
    
    # Validation predictions (valid_predictors_int)
    pred_val <- predict(qrf_model, data = valid_predictors_int, 
                        type = "quantiles", quantiles = 0.5)$predictions
    # print(pred_val)
    # Calculate metrics for calibration (training set)
    rsq_cal <- (cor(pred_cal, train_response_int))^2          # coefficient of determination (using correlation) - R²
    mse_cal <- mean((train_response_int - pred_cal)^2)        # Mean Squared Error -	MSE
    rmse_cal <- sqrt(mean((train_response_int - pred_cal)^2)) # Root Mean Squared Error - RMSE
    mae_cal <- mean(abs(train_response_int - pred_cal))       # Mean Absolute Error - MAE
    # ae_cal <- abs(train_response_int - pred_cal)            # Absolute Error - AE
    # rpd_cal <- sd(train_response_int) / rmse_val            # Ratio of performance to deviation - RPD
    
    # Calculate metrics for validation set (external validation)
    rsq_val <- (cor(pred_val, valid_response_int))^2
    mse_val <- mean((valid_response_int - pred_val)^2)
    rmse_val <- sqrt(mean((valid_response_int - pred_val)^2))
    mae_val <- mean(abs(valid_response_int - pred_val))

    row_index_ls[[i]] <- resample_counter
    Fold_ls[[i]] <- fold_idx
    Resample_ls[[i]] <- resample_name
    rsq_cal_ls[[i]] <- rsq_cal
    mse_cal_ls[[i]] <- mse_cal
    rmse_cal_ls[[i]] <- rmse_cal
    mae_cal_ls[[i]] <- mae_cal
    rsq_val_ls[[i]] <- rsq_val
    mse_val_ls[[i]] <- mse_val
    rmse_val_ls[[i]] <- rmse_val
    mae_val_ls[[i]] <- mae_val
    
    # Increment counter
    resample_counter <- resample_counter + 1
    
    # Progress update
    cat(sprintf("Rep %d, Fold %d \n", r, fold_idx))
  }
}
})[3] # 30 seconds

# format(object.size(models_list), units = "auto") # 2 GB

# Gather results 
results_folds_rep <- data.frame(
  row_index = unlist(row_index_ls),
  Fold = unlist(Fold_ls),
  Resample = unlist(Resample_ls),
  R2_cal = unlist(rsq_cal_ls),
  MSE_cal = unlist(mse_cal_ls),
  RMSE_cal = unlist(rmse_cal_ls),
  MAE_cal = unlist(mae_cal_ls),
  R2_val = unlist(rsq_val_ls),
  MSE_val = unlist(mse_val_ls),
  RMSE_val = unlist(rmse_val_ls),
  MAE_val = unlist(mae_val_ls)
  )
}

head(results_folds_rep)
tail(results_folds_rep)
write.csv(results_folds_rep, 
          file = "../output/results_qrf_model_cv_10_rep_100.csv", 
          row.names = F)

### Metrics average by fold
results_by_fold <- results_folds_rep |> 
  group_by(Fold) |> 
  summarise(
    R2_cal = mean(R2_cal),
    MSE_cal = mean(MSE_cal),
    RMSE_cal = mean(RMSE_cal),
    MAE_cal = mean(MAE_cal),
    R2_val = mean(R2_val),
    MSE_val = mean(MSE_val),
    RMSE_val = mean(RMSE_val),
    MAE_val = mean(MAE_val)
  )
results_by_fold <- sapply(results_by_fold, round, 3)
write.csv(results_by_fold, 
          file = "../output/results_qrf_model_cv_10_rep_100_average_per_fold.csv", 
          row.names = F)

# Select only numeric columns and calculate the mean
model_cv_avg <- colMeans(results_folds_rep[,  4:ncol(results_folds_rep)])
model_cv_avg <- sapply(model_cv_avg, round, 3)
model_cv_avg <- c(Num_folds = as.character(k),
                  Num_repetitions= as.character(repeats), 
                  Total_iterations = as.character(total_iterations), 
                  model_cv_avg)
as.data.frame(t(model_cv_avg))
write.csv(as.data.frame(t(model_cv_avg)), 
          file = "../output/results_qrf_model_cv_10_rep_100_average.csv", 
          row.names = F)

################################################################################
# Train the Quantile Random Forest model with ranger
{
set.seed(123)
qrf_model_f <- ranger::ranger(
  x                 = train_v_predictors,
  y                 = train_v_response,
  num.trees         = num_trees,
  mtry              = mtry,
  min.node.size     = min_node_size,
  sample.fraction   = sampe_size,
  max.depth         = max_depth,
  splitrule         = splitrule,
  # num.random.splits = num_random_splits, # for extratrees only
  importance        = importance,
  quantreg          = TRUE, # Enable Quantile Regression Forest
  replace           = T,
  num.threads       = n_cores 
)

# Calibration predictions (on training set)
pred_cal <- predict(qrf_model_f, data = train_v_predictors, 
                     type = "quantiles", quantiles = 0.5)$predictions
# Calculate metrics for calibration (training set)
(rsq_cal <- (cor(pred_cal, train_v_response))^2)
(mse_cal <- mean((train_v_response - pred_cal)^2))
(rmse_cal <- sqrt(mean((train_v_response - pred_cal)^2)))
(mae_cal <- mean(abs(train_v_response - pred_cal)))

# Test predictions (on test set)
pred_test <- predict(qrf_model_f, data = test_predictors, 
                     type = "quantiles", quantiles = 0.5)$predictions
# Calculate metrics for calibration (training set)
(rsq_test <- (cor(pred_test, test_response))^2)
(mse_test <- mean((test_response - pred_test)^2))
(rmse_test <- sqrt(mean((test_response - pred_test)^2)))
(mae_test <- mean(abs(test_response - pred_test)))
}

results_test <- data.frame(
  R2_cal = rsq_cal,
  MSE_cal = mse_cal,
  RMSE_cal = rmse_cal,
  MAE_cal = mae_cal,
  R2_test = rsq_test,
  MSE_test = mse_test,
  RMSE_test = rmse_test,
  MAE_test = mae_test
)
class(results_test)
results_test <- as.data.frame(t(sapply(results_test, round, 3)))

write.csv(results_test, 
          file = "../output/results_qrf_model_test.csv", 
          row.names = F)

# Save the model object to a file
saveRDS(qrf_model_f, file = "../output/qrf_model_test.rds") 
# Restore the object
# qrf_model_f <- readRDS(file = "../output/dataset/qrf_model_test.rds") 

################################################################################ 
### Plot Calibration and Test
################################################################################
library(tune) # for coord_obs_pred()

################################### Calibration
observed_values <- train_v_response
predicted_values <- pred_cal
max(observed_values)
max(predicted_values)

plot(pred_cal, train_v_response)

# Create a data frame containing both observed and predicted values
plot_data <- data.frame(Observed = observed_values, Predicted = predicted_values[,1])

# Create the ggplot
p <- ggplot(plot_data, aes(x = Predicted, y = Observed)) +
  geom_point(aes(color = "Data Points"), alpha = 0.5) +  # Scatter plot
  geom_smooth(method = 'lm', aes(color = "Fitted Line"), se = F, fullrange = T, size = 0.6) +  # Fitted line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", size = 0.3, color = "black") +  # 1:1 line
  scale_color_manual(values = c("Data Points" = "blue", "Fitted Line" = "darkorange1")) +
  ggtitle("Test") +
  scale_x_continuous(limits = c(0, 10.5)) + # Extend the geom_somth line
  scale_y_continuous(limits = c(0, 10.5)) + # Extend the geom_somth line
  xlab(paste0("Predicted SOC %")) +
  ylab(paste0("Observed SOC %")) +
  annotate("text", x = 5, y = 0.7, 
           label = paste("R² ", round(rsq_cal, 2), "\n",
                         "RMSE ", round(rmse_cal, 2)), parse = F) +
  theme_minimal() + coord_obs_pred() # library(tune)
p

# Save the plot using ggsave
ggsave(filename= paste0("../output/obs_pred_qrf_model_calibration.png"), 
       plot = p, 
       width = 9, height = 9, dpi = 300)


################################### Test
observed_values <- test_response
predicted_values <- pred_test

plot(pred_test, test_response)

# Create a data frame containing both observed and predicted values
plot_data <- data.frame(Observed = observed_values, Predicted = predicted_values[,1])

# Create the ggplot
p <- ggplot(plot_data, aes(x = Predicted, y = Observed)) +
  geom_point(aes(color = "Data Points"), alpha = 0.5) +  # Scatter plot
  geom_smooth(method = 'lm', aes(color = "Fitted Line"), se = F, fullrange = T, size = 0.6) +  # Fitted line
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", size = 0.3, color = "black") +  # 1:1 line
  scale_color_manual(values = c("Data Points" = "blue", "Fitted Line" = "darkorange1")) +
  ggtitle("Test") +
  scale_x_continuous(limits = c(0, 6.5)) + # Extend the geom_somth line
  scale_y_continuous(limits = c(0, 6.5)) + # Extend the geom_somth line
  xlab(paste0("Predicted SOC %")) +
  ylab(paste0("Observed SOC %")) +
  annotate("text", x = 3, y = 0.7, 
           label = paste("R² ", round(rsq_test, 2), "\n",
                         "RMSE ", round(rmse_test, 2)), parse = F) +
  theme_minimal() + coord_obs_pred() # library(tune)
p

# Save the plot using ggsave
ggsave(filename= paste0("../output/obs_pred_qrf_model_test.png"), 
       plot = p, 
       width = 9, height = 9, dpi = 300)

################################################################################
### Plot the importance of covariates
################################################################################
# Load the required package
library(ggplot2)

# data frame for variable importance
var_imp_df <- as.data.frame(qrf_model_f$variable.importance)
var_imp_df$Variable <- rownames(var_imp_df)
rownames(var_imp_df) <- NULL

# Reaname columns to facilitate use in ggplot2
colnames(var_imp_df) <- c("Importance", "Variable")

# Sort by importance
var_imp_df <- var_imp_df[order(-var_imp_df$Importance),]
write.csv(var_imp_df, file = "../output/var_imp_df_qrf_model_test.csv", row.names = F)

library(scales)
var_imp_df$Importance <- scales::rescale(var_imp_df$Importance, to = c(1, 100))
write.csv(var_imp_df, file = "../output/var_imp_df_qrf_model_test_scaled.csv", row.names = F)
# var_imp_df <- var_imp_df[1:10, ]

# Generate the lollipop plot with an improved theme
ggplot(var_imp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_segment(aes(xend = reorder(Variable, Importance), yend = 0), 
               color = "dodgerblue", 
               linewidth = 0.5) +
  geom_point(color = "dodgerblue", size = 3) +
  coord_flip() +
  theme_minimal() +
  theme(
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14, face = "bold")
  ) +
  xlab("Variable") +
  ylab("Importance") +
  ggtitle("Variable Importance")

# Save the plot using ggsave
ggsave(filename= paste0("../output/Covariate_importance_qrf_model_test.png"), 
       plot = last_plot(), 
       width = 7, height = 10, dpi = 300)

################################################################################
### Density plot 
################################################################################
# Each dataset + predicted
# Create a data frame with the SOC values and the dataset
plot_data <- data.frame(
  SOC = c(train_response, valid_response, test_response, pred_test[, 1]),
  Dataset = rep(c("Training", "Validation", "Test", "Predicted"), 
                times = c(length(train_response), 
                          length(valid_response),
                          length(test_response),
                          length(pred_test[, 1]))))

# Medians
median_train <- median(train_response)
median_valid <- median(valid_response)
median_test <- median(test_response)
median_pred <- median(pred_test[, 1])

# Create the plot with histogram and density lines
p <- ggplot(plot_data, aes(x = SOC, fill = Dataset, color = Dataset)) +
  ### Density lines
  geom_density(alpha = 0.2) +
  ### Vertical dashed lines for the medians
  geom_vline(aes(xintercept= median_train, color= "Training"), linetype= "dashed", linewidth= 0.5) +
  geom_vline(aes(xintercept= median_valid, color= "Validation"), linetype= "dashed", linewidth= 0.5) +
  geom_vline(aes(xintercept= median_test, color= "Test"), linetype= "dashed", linewidth= 0.5) +
  geom_vline(aes(xintercept= median_pred, color= "Predicted"), linetype= "dashed", linewidth = 0.5) +
  ### Colors
  scale_fill_manual(
    values = c(
      "Training" = "lightgoldenrod3",  # Light goldenrod
      "Validation" = "lightgreen",     # Light green
      "Test" = "lightblue",            # Light blue
      "Predicted" = "lightcoral"       # Light coral
    ),
    labels = c("Training", "Validation", "Test", "Predicted")
  ) +
  scale_color_manual(
    values = c(
      "Training" = "lightgoldenrod3",
      "Validation" = "lightgreen",
      "Test" = "lightblue",
      "Predicted" = "lightcoral"
    ),
    labels = c("Training", "Validation", "Test", "Predicted")
  ) +
  # Title and labels
  labs(title = "Density of SOC Across Datasets",
       # subtitle = "",
       caption = "Dashed lines are median") +
  xlab("SOC (%)") +
  ylab("Density") +
  # Theme and adjustments for publication
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    # legend.title = element_blank(),  # Remove the legend title
    legend.title = element_text("Datasets"),
    legend.text = element_text(size = 10),
    legend.position.inside = c(0.9, 0.9),
    legend.background = element_rect(fill = "white", color = NA)
  )
p
# # Save the plot
ggsave(filename= paste0("../output/Datasets_density.png"),
       plot = p,
       width = 9, height = 6, dpi = 250)

######################## Train + Valid, test, pred
# Create a data frame with the SOC values and the dataset
plot_data <- data.frame(
  SOC = c(train_response, valid_response, test_response, pred_test[, 1]),
  Dataset = rep(c("Train-CV", "Test", "Predicted"), 
                times = c(length(c(train_response, valid_response)), 
                          length(test_response),
                          length(pred_test[, 1]))))

# Medians
median_train <- median(c(train_response, valid_response))
median_test <- median(test_response)
median_pred <- median(pred_test[, 1])

# Create the plot with histogram and density lines
p <- ggplot(plot_data, aes(x = SOC, fill = Dataset, color = Dataset)) +
  ### Density lines
  geom_density(alpha = 0.2) +
  ### Vertical dashed lines for the medians
  geom_vline(aes(xintercept= median_train, color= "Train-CV"), linetype= "dashed", linewidth= 0.5) +
  # geom_vline(aes(xintercept= median_valid, color= "validation"), linetype= "dashed", linewidth= 0.5) +
  geom_vline(aes(xintercept= median_test, color= "Test"), linetype= "dashed", linewidth= 0.5) +
  geom_vline(aes(xintercept= median_pred, color= "Predicted"), linetype= "dashed", linewidth = 0.5) +
  ### Colors
  scale_fill_manual(
    values = c(
      "Train-CV" = "lightgreen",       # Light goldenrod
      # "validation" = "lightgoldenrod3", # Light green
      "Test" = "lightblue",               # Light blue
      "Predicted" = "lightcoral"          # Light coral
    ),
    labels = c("Train-CV", 
               # "validation", 
               "Test", "Predicted")
  ) +
  scale_color_manual(
    values = c(
      "Train-CV" = "lightgreen",
      # "validation" = "lightgoldenrod3",
      "Test" = "lightblue",
      "Predicted" = "lightcoral"
    ),
    labels = c("Train-CV", 
               # "validation", 
               "Test", "Predicted")
  ) +
  # Title and labels
  labs(title = "Density of SOC Across Datasets",
       # subtitle = "",
       caption = "Dashed lines are median") +
  xlab("SOC (%)") +
  ylab("Density") +
  # Theme and adjustments for publication
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, hjust = 0.5),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text("Datasets"),
    legend.text = element_text(size = 10),
    legend.position.inside = c(0.9, 0.9),
    legend.background = element_rect(fill = "white", color = NA)
  )
p
# Save the plot
ggsave(filename= paste0("../output/Datasets_density_Train-Valid_test_pred.png"), 
       plot = p, 
       width = 9, height = 6, dpi = 250)


################################################################################
### Predict tifs, map and plot 
################################################################################
library(terra)
library(ranger)
library(parallel)
qrf_model_f <- readRDS(file = "../output/dataset/qrf_model_test.rds") 

covariates <- terra::rast("../../1_Covariates_preparation/output/Covariates_Kenya.tif")
# names(covariates)

# Check NA
global(covariates, fun = "isNA", na.rm = FALSE)

# Replace with 0
covariates <- subst(covariates, NA, 0)

dir.create("../output/temp", showWarnings = FALSE)
terraOptions(memmax = 30, tempdir = "../output/temp") 

# use the predict function from ranger package
predfun <- function(...) predict(...)$predictions 
# https://github.com/rspatial/terra/issues/1448
# https://stackoverflow.com/a/77378911/14361772

cls <- parallel::makeCluster(detectCores())
parallel::clusterExport(cls, c("qrf_model_f", "predfun", "ranger"))

# Predict lower bount 0.05
system.time({
raster_pred0.05 <- terra::predict(
  object = covariates,
  model = qrf_model_f,
  fun = predfun,
  type = "quantiles",
  quantiles = 0.05,
  na.rm = TRUE,
  filename = "../output/tif_SOC_mount_Kenya_qrf_model_raster_pred0.05.tif",
  overwrite = TRUE,
  cores = 12, 
  cpkgs = "ranger",  # Export the ranger package to all nodes
  wopt = list(
    gdal = c("COMPRESS=LZW", "TFW=YES"),
    progress = TRUE
  )
)
})[3] # 5min
terra::plot(raster_pred0.05)

# Predict median bount 0.5
system.time({
  raster_pred0.50 <- terra::predict(
    object = covariates,
    model = qrf_model_f,
    fun = predfun,
    type = "quantiles",
    quantiles = 0.5,
    na.rm = TRUE,
    filename = "../output/tif_SOC_mount_Kenya_qrf_model_raster_pred0.50.tif",
    overwrite = TRUE,
    cores = 12, 
    cpkgs = "ranger", 
    wopt = list(
      gdal = c("COMPRESS=LZW", "TFW=YES"),
      progress = TRUE
    )
  )
})[3] # 5min
terra::plot(raster_pred0.50)

# Predict upper bount 0.95
system.time({
  raster_pred0.95 <- terra::predict(
    object = covariates,
    model = qrf_model_f,
    fun = predfun,
    type = "quantiles",
    quantiles = 0.95,
    na.rm = TRUE,
    filename = "../output/tif_SOC_mount_Kenya_qrf_model_raster_pred0.95.tif",
    overwrite = TRUE,
    cores = 12, 
    cpkgs = "ranger", 
    wopt = list(
      gdal = c("COMPRESS=LZW", "TFW=YES"),
      progress = TRUE
    )
  )
})[3] # 5min
terra::plot(raster_pred0.95)

parallel::stopCluster(cls)

# Compute range as uncertainties
range_interval_0.9 <- raster_pred0.95 - raster_pred0.05

uncertainties <- c(raster_pred0.05, raster_pred0.95, raster_pred0.50, range_interval_0.9)
# names(uncertainties) <- c("lower_bound", "upper_bound", "mean_SOC", "range_interval")
names(uncertainties) <- c("lower_bound", "upper_bound", "median_SOC", "range_interval")

# Save the created maps
writeRaster(range_interval_0.9, "../output/tif_SOC_mount_Kenya_qrf_model_raster_pred_range_0.05_0.95.tif",
            overwrite=TRUE,
            wopt = list(
              gdal = c("COMPRESS=LZW", "TFW=YES"),
              progress = TRUE))

### Re-load the tif images
library(terra)
lower_bound <- terra::rast("../output/maps/tif_SOC_mount_Kenya_qrf_model_raster_pred0.05.tif")
upper_bound <- terra::rast("../output/maps/tif_SOC_mount_Kenya_qrf_model_raster_pred0.95.tif")
median_SOC <- terra::rast("../output/maps/tif_SOC_mount_Kenya_qrf_model_raster_pred0.50.tif")
range_interval <- terra::rast("../output/maps/tif_SOC_mount_Kenya_qrf_model_raster_pred_range_0.05_0.95.tif")

uncertainties <- c(lower_bound, upper_bound, median_SOC, range_interval)
names(uncertainties) <- c("lower_bound", "upper_bound", "median_SOC", "range_interval")

################################################################################
### Make maps
################################################################################
for (j in 1:4) {
  # Plot the map
  terra::plot(uncertainties[[j]], 
              main = names(uncertainties)[j], 
              col = c("white", "grey", "grey20", "black"),
              legend = T)
  Sys.sleep(1)
}

col_pall <- colorRampPalette(c("#f1d8ba", "#c4a379", "#96744d", "#6b4923", "#654321"))
col_pall2 <- colorRampPalette(c("#00FFFF", "#0000FF", "#800080", "#FF0000", "#8B0000"))

ib <- ext(covariates)

### Plot and save in loop
for (k in 1:4) {
  
  cat("Iteration: ", k, "of 4 \n")
  cat("Map: ", names(uncertainties)[k], "\n")
  
  # Save the plot    
  png(filename = paste0("../output/", names(uncertainties[[k]]), "_soc_map_uncertainties.png"),
      width = 20, height = 20, units = "cm",
      pointsize = 12, bg = "white",  res = 300)
  
  # Run the plot
  terra::plot(uncertainties[[k]], # type= "interval", 
              main= paste0(names(uncertainties[[k]])), # " SOC (%) map"),
              plg=list( # for legend
                title=" SOC (%)",
                title.cex=0.8,  # legend title font size
                shrink=0.8, 
                leg.shrink=0.8,
                leg.width=1,  
                cex=0.9), # legend font size
              mar= c(3.1, 4.1, 3.1, 7.1), 
              box= T, axes= F, smooth= T,
              xlim = c(ib[1], ib[2]), ylim = c(ib[3], ib[4]), # cut for inner box 
              # col = col_pall(100), # change for switch in each map
              col = switch(k, 
                           col_pall2(20), 
                           col_pall2(20), 
                           col_pall(20), 
                           col_pall2(20))
  )
  terra::north(type= 1, cex = 1, xy=c(ib[1]+0.52, ib[3]-0.03), xpd= T)
  terra::sbar(10, lonlat = T, xy=c(ib[1]+0.38, ib[3]-0.04), labels = c("0", " ", "10 km"),
              adj=c(0.5, -1.2), type= "bar", divs= 3, xpd= T) # xy="bottomright" lab = '1 km'
  
  # Sys.sleep(1)
  dev.off()
}


################################################################################
### Calculate the PICP 
################################################################################
# https://www.r-bloggers.com/2023/08/calculating-the-prediction-interval-coverage-probability-picp/

library(tune)
calcPICP = function(data, response, pred){
  
  # We first get the residuals of the model
  res = response - pred
  
  # Then we get the standard deviation of the residuals and combine with the data.
  data$stdev = sd(res)
  
  # We than make a series of quantiles from a normal cumulative distribution.
  qp <- qnorm(c(0.995, 0.9875, 0.975, 0.95, 0.9, 0.8, 0.7, 0.6, 0.55, 0.525))
  
  # Then make a matrix the with the row length of the data and columns of qp
  vMat <- matrix(NA, nrow = nrow(data), ncol = length(qp))
  
  # Now we must loop around the quantiles and multiply it by the standard deviation to get a series of standard errors with different prediction intervals. 
  for(i in 1:length(qp)){
    vMat[,  i] <- data$stdev * qp[i]
  }
  
  # Make another matrix same as before for the upper limits
  uMat <- matrix(NA, nrow = nrow(data), ncol = length(qp))
  
  # We calculate the upper limits by adding the series of standard errors to the predictions of the model. 
  for(i in 1:length(qp)) {
    uMat[,  i] <- pred + vMat[, i]
  }
  
  # We make another matrix for the lower limits
  lMat <- matrix(NA, nrow = nrow(data), ncol = length(qp))
  
  # We calculate the lower limits by subtracting the series from the predicted values.
  for(i in 1:length(qp)) {
    lMat[, i] <- pred - vMat[,  i]
  }
  
  # Now we want to see which prediction intervals cover the measured data creating a matrix of 1s and 0s. 
  bMat <- matrix(NA, nrow = nrow(data), ncol = length(qp))
  
  for(i in 1:ncol(bMat)){
    bMat[, i] <- as.numeric(response <= uMat[,  i]  &
                              response >= lMat[, i])
  }
  
  # To calculate the PICP we take the colsums/nrow*100 for the matrix of 0s and 1s
  picp <- colSums(bMat)/nrow(bMat)*100
  
  # Make a vector of confidence levels
  cl <- c(99, 97.5, 95, 90, 80, 60, 40, 20, 10, 5)
  
  # We put into a data frame for plotting
  results <- data.frame(picp = picp, cl = cl)
  
  # Since we want PICP to CI to be a 1:1 line we also calculate Lin’s concordance correlation coefficient (CCC) with the yardstick R package.
  ccc <- as.data.frame(yardstick::ccc_vec(results$picp, results$cl))
  
  # Make name correct
  names(ccc) = "CCC" #name
  
  # must add axis values for plotting
  ccc$x = 12 #x axis
  ccc$y = 90 #y axis
  
  # Now we can plot the PICP to CI, add the 1:1 line and the CCC
  p = ggplot(data = results, aes(x= cl, y = picp)) + #add data
    geom_point()+ #add points
    geom_text(data = ccc, aes(x= x, y =y, label = paste("CCC = ", round(CCC, 2))))+ # add CCC value
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = 'red')+ # add 1:1 line
    labs(x = 'Confidence level', y = 'PICP', title = 'PICP to confidence level')+ # labels
    # coord_fixed(ratio=1) +
    tune::coord_obs_pred() + # make the plot square # https://tune.tidymodels.org/reference/coord_obs_pred.html
    theme_bw() #make it look good
  
  # Now we want to return a list of the plot as well as a data frame of the total results.
  return(setNames(list(p, results), c("Plot", "Results")))
}

# Now we have the function giving us a plot of the PICP to CI and results. This is useful when running many models and now we can just plug in the data.

# get the values to input in the function
# get predictions
pred_test <- predict(qrf_model_f, data = test_predictors, 
                     type = "quantiles", quantiles = 0.5)$predictions
# dat <- cbind(test_predictors, pred = pred_test[, 1]) # combine the data
dat <- as.data.frame(cbind(SOC= test_response, pred = pred_test[, 1])) # combine the data

# picp = picpCalc(dat, dat$clay, dat$pred)
picp = calcPICP(dat, dat$SOC, dat$pred)

# now, plot the data.
picp[1]
picp[2]

ggsave(filename= paste0("../output/picp_plot.png"), 
       plot = picp[[1]], 
       width = 9, height = 9, dpi = 300)


################################################################################
# Example data 
dat_matrix <- matrix(c(
  1.674, 1.915998,
  3.262, 3.703449,
  2.442, 2.340370,
  1.842, 2.627656,
  1.917, 2.039686,
  1.072, 1.722476,
  0.999, 1.421247,
  1.499, 1.792844,
  0.921, 1.795698,
  1.747, 1.763379,
  1.521, 2.149659,
  2.323, 2.363802
), byrow = TRUE, ncol = 2)

dat <- as.data.frame(dat_matrix)
colnames(dat) <- c("SOC", "pred")

dat
picp = calcPICP(dat, dat$SOC, dat$pred)
