#############################################################################################
### Mapping Mount Kenya Region 
### Yuri Andrei Gelsleichter
### May 2024 
#############################################################################################

### Set working directory automatically 
library(this.path)
path_ <- this.dir()
setwd(path_)

### Check the "new" working directory 
getwd()

### clean memory
# gc(); rm(list=ls())

library(terra) # for rasters and vectors
# library(dplyr) # for distinct function

### Load the data
dir("../../")
dir("../../0_Data_source_metadata_data_treatement")
dir("../../0_Data_source_metadata_data_treatement/output")
sample_points <- terra::vect("../../0_Data_source_metadata_data_treatement/output/sample_points_df.shp")
terra::plot(sample_points)
sample_points
terra::ext(sample_points)
terra::ext(terra::buffer(sample_points, 10 * 1000)) # 10 km buffer square to input in gee

dir("../../1_Covariates_preparation/")
covariates <- terra::rast("../../1_Covariates_preparation/output/Covariates_Kenya.tif")

names(covariates)
names(covariates[[4]]) # landform
names(covariates[[8]]) # landcover

# Convert categorical variables (layers) to factors
# see: https://rdrr.io/github/rspatial/terra/man/factors.html
covariates[[4]] <- as.factor(round(covariates[[4]], 0)) # landforms
covariates[[7]] <- terra::as.int(covariates[[7]]) # landcover, convert integer first
covariates[[8]] <- terra::as.factor(covariates[[8]]) # landcover, convert to factor (discrete value)

# Verify the categorical levels
levels(covariates)

# plot covariates
terra::plot(covariates[[1:6]])
terra::plot(covariates[[7:12]])
terra::plot(covariates[[11]]) # flow accumulation
terra::plot(covariates[[13:18]])
terra::plot(covariates[[19:22]])
terra::plot(covariates[[53]])

terra::plot(covariates[[2]])
terra::plot(log(covariates[[2]]))
terra::plot(log(abs(covariates[[2]])))
terra::plot(sample_points, add= T)

library(viridis)

# Visual check in each layer (bit heavy)
names(covariates) |> as.data.frame()
system.time(
for (i in 1:length(names(covariates))) {
# for (i in 1:2) {
  # Loop status
  cat("Plot covariate layer: ", names(covariates[[i]]), "\n",
      i, " of ", length(names(covariates)), "\n")
  
  png(filename = paste0("../output/covariates_plot_png/Covariate_",
                        names(covariates[[i]]), ".png"), units = "px", pointsize = 12,
      width = 900, height = 1100, res = 150, bg = "transparent")
  if (i %in% c(1:2, 5, 7, 9, 13:19, 22:27, 41:43, 48, 53:54)) { 
    # terra::plot(log(abs(covariates[[i]])), main = paste0(names(covariates[[i]]), " - \nrescale with log(abs())"), legend= T, axes= F)
    terra::plot(log(abs(covariates[[i]])), main = paste0(names(covariates[[i]])), 
                legend= T, axes= F, col= viridis(200))
    # mtext("rescale with log(abs())", side = 3, line = -0.7, cex = 0.6)
    mtext("Few outliers pixels were driving the scale out and the image had only one value, \nrescale with log(abs()) to adjust it and present the image features", 
          side = 1, line = 1, cex = 0.6)
  } else {
    terra::plot(covariates[[i]], main = names(covariates[[i]]), 
                legend= T, axes= F, col= viridis(200))
  }
  # terra::plot(sample_points, border= "skyblue", add = T, lwd = 1)
  # terra::plot(sample_points, col= "green4", add = T)
  dev.off()
  # Sys.sleep(0.1)
}
)[3] # 10 min

points_df <- read.csv("../../0_Data_source_metadata_data_treatement/output/sample_points_df.csv")

### With Terra package
### Extract point location values from covariates
system.time({df.mod = terra::extract(covariates, 
                                      sample_points, 
                                      bind = T, # if true return a shapefile with all columns
                                      method= "simple", 
                                      # "bilinear" (bilinear return values from four nearest raster cells, but is also interpolate categorical variables, messing with caterical variables) 
                                      # "simple" preserve the caterical variables 
                                      xy= T, 
                                      ID= F)})
df.mod
(df.mod[, 1:12])
(df.mod[, 12:22])

# df.mod <- terra::as.data.frame(df.mod, geom="XY") # geom="XY" return coordinates, but the df already has, than duplicate 
df.mod <- terra::as.data.frame(df.mod)

str(df.mod)

################################################################################
### Cov selection: correlation 
################################################################################
################################################################################
##### Determine highly correlated variables in function of predicted soil property  
# Determine the covariates using "pearson" (default) correlation
################################################################################
# install.packages("corrplot")
library(corrplot)
names(df.mod[, c(2:ncol(df.mod))]) # this can detect multicolinearity, full data SOC correlation

# drop categorical variables for correlation 
dput(names(df.mod))
df_mod_corr <- subset(df.mod, select = -c(Soil.ID, data_type, landform, landcover))

### Transform categorical variables Geology and Geomorfology in variables indicators
# df.mod.ind <- model.matrix(object = ~. - 1, data= df.mod[, c(2:ncol(df.mod))])
# df.mod.ind
# M = cor(df_mod_corr, use = "complete.obs")

library(Hmisc)
cor_result <- Hmisc::rcorr(as.matrix(df_mod_corr), type = "pearson")  # can be "spearman"

# Get correlation matrix
M <- cor_result$r

# Get p-value matrix
p.mat <- cor_result$P

# save correlation plot
png(height=4200, width=4200, res= 200,
file= "../output/graphs_plots/covariates_correlation_sig.level.png")

corrplot(M, 
         order = 'original', 
         addCoef.col = 'black', 
         number.cex = .7, 
         tl.cex = .8, 
         number.font = 1, 
         number.digits = 2, 
         tl.srt = 45, 
         title = "Covariates correlation",
         type = "full", 
         cl.pos = 'n', 
         col = COL2('RdYlBu'),
         mar = c(0, 0, 2, 0),
         p.mat = p.mat,              # Matriz de valores-p
         sig.level = 0.1,           # Nível de significância (0.05)
         insig = "pch",              # Marcar valores insignificantes com 'X'
         pch = 4,                    # Símbolo 'X' para valores insignificantes
         pch.cex = 1.2)              # Tamanho do símbolo
dev.off()

# save correlation plot
png(height=4200, width=4200, res= 200,
file= "../output/graphs_plots/covariates_correlation.png")

corrplot(M, order = 'original', addCoef.col = 'black', #tl.pos = 'd', 
         number.cex = .7, tl.cex = .8, number.font = 1, number.digits = 2,
         tl.srt = 45, 
         title = "Covariates correlation",
         type = "full", cl.pos = 'n', col = COL2('RdYlBu'),
         mar = c(0, 0, 2, 0)
)
dev.off()

### With "fst" faster and lighter
# install.packages("fst")
library(fst)
system.time({ # 5 seconds to write and 100 MB
  write_fst(df.mod, "../output/dataset_mod_22_covariates_df.fst", compress = 100)
})

# system.time({ # .5 second to read
# df.pred <- read_fst("../output/dataset/dataset_mod_22_covariates_df.fst")
# })

# Source: 
# https://data.nozav.org/post/2019-r-data-frame-benchmark/
# https://edomt.github.io/File-IO-Storage/ # faster and lighter
# https://waterdata.usgs.gov/blog/formats/ # very good
# https://community.rstudio.com/t/compres-data-to-export-it/48000/2

"
### Data source
https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001
https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_mTPI 
https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_CHILI/bands
https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_landforms
https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_topoDiversity
https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1/
https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200
https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_MONTHLY
https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2 
"

