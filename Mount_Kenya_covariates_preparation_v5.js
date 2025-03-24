////////////////////////////////////////////////////////////////////////////////
/* 
Covariates preparation for Kenya 
Authors: Yuri Gelsleichter
May 2025
*/
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Clip images are unecessary and should be avoided, or done ate end of script, references: 
// https://developers.google.com/earth-engine/guides/best_practices?hl=en#if-you-dont-need-to-clip,-dont-use-clip
// https://courses.spatialthoughts.com/end-to-end-gee.html (item 07)
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Copernicus DEM from GEE ImageCollection :: https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_DEM_GLO30 //// 
////////////////////////////////////////////////////////////////////////////////
// Does not perform the terrain derivations // var dem_ic = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM');

////////////////////////////////////////////////////////////////////////////////
/// NASADEM: NASA NASADEM Digital Elevation 30m :: https://developers.google.com/earth-engine/datasets/catalog/NASA_NASADEM_HGT_001
////////////////////////////////////////////////////////////////////////////////
var dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation').rename('dem');

// Get scale from DEM (30.922080775909325 m) 
var dem_scale = dem.projection().nominalScale();
print('DEM resolution (before mosaic)', dem_scale); 

// print("DEM", dem_ic); // too large, too many imiages to get info, console breaks
// print("DEM", dem_ic.limit(100)) // limnit images 

// Create Mosaic 
//// var dem = dem.mosaic().rename('dem'); // Only neede with copernicus dem 
// print("DEM", dem); 

// Get projection, the default projection is WGS84 with 1-degree scale 
var dem_crs = dem.projection();
// Check projection 
print('DEM crs', dem_crs);

var elevationVis = {
  min: 0.0,
  max: 1000.0,
  palette: ['0000ff','00ffff','ffff00','ff0000','ffffff'],
};
Map.setCenter(37.8, -0.10, 9); // x and y, longitude and latitude, zoom level 
Map.addLayer(dem, elevationVis, 'DEM', false);

// Calculate slope. Units are degrees, range is [0,90).
var slope = ee.Terrain.slope(dem).rename('slope');

// Calculate aspect. Units are degrees where 0=N, 90=E, 180=S, 270=W.
var aspect = ee.Terrain.aspect(dem).rename('aspect');

// Calculate Hillshade
// var hillshade = ee.Terrain.hillshade(dem).rename('hillshade');

// Hillshade image from a DEM :: https://gis.stackexchange.com/questions/445241/hillshade-image-blend-that-is-not-washed-out-earth-engine
var hillshade = ee.Terrain.hillshade({
  // Divide the DEM by 2 to flatten it a little, otherwise the terrain effect is
  // too exagerated - play around with how much you flatten it.
  // input: dem.divide(2),
  input: dem, 
  azimuth: 270,
  elevation: 45
}).rename('hillshade');
// ee.Terrain.hillshade(input, azimuth, elevation)
// Argument	Type	Details
// input	Image	An elevation image, in meters.
// azimuth	Float, default: 270	The illumination azimuth in degrees from north.
// elevation	Float, default: 45	The illumination elevation in degrees.

// HillShadow image from a DEM
// Does not compute
// var hillShadow = ee.Terrain.hillShadow({
//   image: dem, 
//   azimuth: 270, 
//   zenith: 45, 
//   neighborhoodSize: 4, 
//   hysteresis: false
// }).rename('hillShadow');

// Calculate Northernness 
var northernness = aspect.subtract(180).abs().rename('northernness');

/*
Map.addLayer(slope, {min: 0, max: 89.99}, 'Slope', false);
Map.addLayer(aspect, {min: 0, max: 359.99}, 'Aspect', false);
// Map.addLayer(hillShadow, {min: 0, max: 255}, 'HillShadow', false);
Map.addLayer(hillshade, {min: 0, max: 255}, 'Hillshade', false);
Map.addLayer(northernness, {min: 0, max: 359.99}, 'Northernness', false);
*/

// OR: 
// Use the ee.Terrain.products function to calculate slope, aspect, and
// hillshade simultaneously. The output bands are appended to the input image.
// Hillshade is calculated based on illumination azimuth=270, elevation=45.
// var terrain = ee.Terrain.products(dem);
// print('ee.Terrain.products bands', terrain.bandNames());
// Map.addLayer(terrain.select('slope'), {min: 0, max: 255}, 'Slope');
// Map.addLayer(terrain.select('aspect'), {min: 0, max: 255}, 'Aspect');
// Map.addLayer(terrain.select('hillShadow'), {min: 0, max: 255}, 'HillShadow');
// Map.addLayer(terrain.select('hillshade'), {min: 0, max: 255}, 'Hillshade');
// Map.addLayer(terrain.select('northernness'), {min: 0, max: 255}, 'Northernness');


//////////////////////////////////////////////////////////////// TPI 
//// Global ALOS mTPI (Multi-Scale Topographic Position Index) :: https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_mTPI 
var tpi_im = ee.Image('CSP/ERGo/1_0/Global/ALOS_mTPI').select('AVE').rename('tpi');
// Resolution 
// print('TPI resolution (before mosaic)', tpi_im.projection().nominalScale()); // here .first() cannot be used 
// Projection 
// print('TPI projection', tpi_im.projection());

// Resampling from 270 t0 30 m
var tpi = tpi_im.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution 
print('TPI resolution (30 m)', tpi.projection().nominalScale()); // here .first() cannot be used 

//////////////////////////////////////////////////////////////// CHILI
/// Global ALOS CHILI (Continuous Heat-Insolation Load Index) :: https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_CHILI
// https://www.csp-inc.org/
// https://mw1.google.com/ges/dd/images/CSP_ERGo_CHILI_sample.png
var chili = ee.Image('CSP/ERGo/1_0/Global/ALOS_CHILI').select('constant').rename('chili');

// Resampling from 90 t0 30 m
chili = chili.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('chili resolution', chili.projection().nominalScale()); 

//////////////////////////////////////////////////////////////// Landforms
/// Global ALOS Landforms :: https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_landforms
var landform = ee.Image('CSP/ERGo/1_0/Global/ALOS_landforms').rename('landform');

// Resampling from 90 t0 30 m
landform = landform.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('landform resolution', landform.projection().nominalScale()); 

//////////////////////////////////////////////////////////////// topo_diver
/// Global ALOS Topographic Diversity :: https://developers.google.com/earth-engine/datasets/catalog/CSP_ERGo_1_0_Global_ALOS_topoDiversity
var topo_diver = ee.Image('CSP/ERGo/1_0/Global/ALOS_topoDiversity').rename('topo_diver');

// Resampling from 270 t0 30 m
topo_diver = topo_diver.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('topo_diver resolution', topo_diver.projection().nominalScale()); 


//////////////////////////////////////////////////////////////// 
/// MERIT Hydro: Global Hydrography Datasets :: https://developers.google.com/earth-engine/datasets/catalog/MERIT_Hydro_v1_0_1/
// Flow Direction (Local Drainage Direction)
var flow_dir = ee.Image('MERIT/Hydro/v1_0_1').select('dir').rename('flow_dir');

// Resampling from 90 t0 30 m
flow_dir = flow_dir.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('flow_dir resolution', flow_dir.projection().nominalScale()); 

var flow_accumul = ee.Image('MERIT/Hydro/v1_0_1').select('upa').rename('flow_accumul');
// Upstream drainage area (flow accumulation area)
// Resampling from 90 t0 30 m
flow_accumul = flow_accumul.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('flow_accumul resolution', flow_accumul.projection().nominalScale()); 

var river_chann_width = ee.Image('MERIT/Hydro/v1_0_1').select('viswth').rename('river_chann_width');
// Visualization of the river channel width
// Resampling from 90 t0 30 m
river_chann_width = river_chann_width.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('River channel width resolution', river_chann_width.projection().nominalScale()); 

////////////////////////////////////////////////////////////////////////////////
/// ESA WorldCover 10m v200 :: https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200
////////////////////////////////////////////////////////////////////////////////
var landcover = ee.ImageCollection('ESA/WorldCover/v200').first().rename('landcover');
// var dataset = ee.ImageCollection('ESA/WorldCover/v100'); 
// var dataset = ee.Image('COPERNICUS/Landcover/100m/Proba-V-C3/Global/2019'); 
// var dataset = ee.Image('ESA/GLOBCOVER_L4_200901_200912_V2_3'); 

// Agregate from 10 to 30 m
landcover = landcover.reduceResolution({
  reducer: ee.Reducer.mean(),
  bestEffort: false // true: attempts to perform the operation as efficiently as possible, even if it means not including all pixels 
}).reproject({
  crs: dem_crs,
  scale: dem_scale
});

// Check resolution
print('landcover resolution', landcover.projection().nominalScale()); 

////////////////////////////////////////////////////////////////////////////////
/// Climate data
////////////////////////////////////////////////////////////////////////////////
// Load the WorldClim Climatology V1 - temperature and preciptation
// https://developers.google.com/earth-engine/datasets/catalog/WORLDCLIM_V1_MONTHLY
var clim = ee.ImageCollection('WORLDCLIM/V1/MONTHLY');

// Select temp_avg from clim
var temp_avg = clim.select('tavg').first().multiply(0.1).rename('temp_avg'); 

// Resampling from 1000 t0 30 m
temp_avg = temp_avg.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('temp_avg resolution', temp_avg.projection().nominalScale()); 


// Select prec from clim
var preciptation = clim.select('prec').first().rename('preciptation'); // .multiply(0.1); 

// Resampling from 1000 t0 30 m
preciptation = preciptation.resample('bilinear')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
                          .reproject({
                            crs: dem_crs,    // EPSG:4326
                            scale: dem_scale // 30 m
                          });

// Check resolution
print('Preciptation resolution', preciptation.projection().nominalScale()); 


////////////////////////////////////////////////////////////////////////////////////
/// Terrain derivations with TAGEE
////////////////////////////////////////////////////////////////////////////////////
var TAGEE = require('users/zecojls/TAGEE:TAGEE-functions');

// Smoothing filter
var gaussianFilter = ee.Kernel.gaussian({
  radius: 3, sigma: 2, units: 'pixels', normalize: true
});

// Smoothing the DEM with the gaussian kernel.
var dem_sm = dem.convolve(gaussianFilter).resample("bilinear");

// Terrain analysis
var terr_attributes = TAGEE.terrainAnalysis(TAGEE, dem_sm); //.updateMask(waterMask); 
// print('Derived Terrain Attributres with TAGEE', terr_attributes.bandNames());

// Apply selected                
var bands_sel = [//"Elevation",
                 "Slope",
                 "Aspect",
                 // "Hillshade",
                 "Northness",
                 "Eastness",
                 "HorizontalCurvature",
                 "VerticalCurvature",
                 "MeanCurvature",
                 "GaussianCurvature",
                 "MinimalCurvature",
                 "MaximalCurvature",
                 "ShapeIndex"
                 ];

// Select the bands to use
var terr_attrib = terr_attributes.select(bands_sel);
print('Selected Derived Terrain Attributres with TAGEE', terr_attrib.bandNames());

/*
/////////////////////////////////////////////////////////////// 
// Load the Landsat 8 image collection 2014-01-01', '2016-12-31
// https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2 
// var Landsat = ee.ImageCollection('LANDSAT/LC09/C02/T2_L2'); // Landsat 9 
// var Landsat = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"); // For Landsat 9 see: https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2

var Landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') // get the image collection 
    // .filterDate('2014-01-01', '2023-12-31')  // Adjust the year interval (done below in calendar range)
    .filter(ee.Filter.lte('CLOUD_COVER', 0.1));  // Filter cloud by percent 

// 2 % cloud + jun, july and aug (6-8)
// or
// 8 % cloud + july and aug (7-8)

// Get scale from Landsat (30 m) 
var Landsat_scale = Landsat.first().projection().nominalScale();
print('Landsat resolution (before mosaic)', Landsat_scale); 
 
// Apply scale factors (to make image with reflectance values and temperature for thermal images)
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

Landsat = Landsat.map(applyScaleFactors);

///---/ No need to filter season, beacuse the cloud filter was more strict /---/// 
// var Landsat = Landsat.filter(ee.Filter.calendarRange(2014,2023,'year'))
//     .filter(ee.Filter.calendarRange(1,2,'month')); // 
// Dry season: Jan to Feb and Jul to Oct :: https://gis.stackexchange.com/a/256334/178680

// Select only the necessary bands 
Landsat = Landsat.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']); 

// reduce image collection with median()
Landsat = Landsat.median();
*/

/*
// Agregate from 30 to 30.922080775909325 m (not suitable for this case)
var Landsat = Landsat.reduceResolution({
  reducer: ee.Reducer.mean(),
  bestEffort: false // true: attempts to perform the operation as efficiently as possible, even if it means not including all pixels 
}).reproject({
  crs: dem_crs,
  scale: dem_scale
});
*/

// // Resampling from 30 to 30.922080775909325 m
// Landsat = Landsat.resample('bicubic')        // bilinear is better to avoid artifactis, bicubic is better but can introduce artifacts 
//                           .reproject({
//                             crs: dem_crs,    // EPSG:4326
//                             scale: dem_scale // 30 m
//                           });


/*
Landsat = Landsat.setDefaultProjection(Landsat.projection());

// // Get the forest cover data at MODIS scale and projection.
// Landsat = Landsat
//     // Request the data at the scale and projection of the MODIS image.
//     .reproject({
//       crs: dem_crs//, 
//       //scale: dem_scale
//     })
//     // Force the next reprojection to aggregate instead of resampling.
//     .reduceResolution({
//       reducer: ee.Reducer.mean() //,
//       //maxPixels: 1024
//     });
// 
//// Get the forest cover data at MODIS scale and projection.
//Landsat = Landsat
//    // Force the next reprojection to aggregate instead of resampling.
//    .reduceResolution({
//      reducer: ee.Reducer.mean(),
//      maxPixels: 1024
//    })
//    // Request the data at the scale and projection of the MODIS image.
//    .reproject({
//      crs: dem_crs//, 
//      //scale: dem_scale
//    });

var Landsat_scale = Landsat.projection().nominalScale();
print('Landsat resolution reprojected', Landsat_scale); 

// Mosaic images 
// var Landsat = Landsat.mosaic();

// Reproject the mosaic to force the resolution to 30 meters after the mosaic (thius step seems not necessary) 
// var Landsat = Landsat.reproject({
//  crs: 'EPSG:4326', // CRS can be adjuted 
//  scale: 30
//});

// Check Landsat
print("See landsat details", Landsat);

var visualization = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0.0,
  max: 0.3,
};

// Add to the map
Map.addLayer(Landsat, visualization, 'True Color (432) Mosaic', false);
*/


////////////////////////////////////////////////////////////////////////////////////
/// bbox, area to clip the rasterrs 
////////////////////////////////////////////////////////////////////////////////////

// var bbox = ee.FeatureCollection("users/gelsleichter/bbox_shp_buff_2km"); // imported shapefile

// 37.5655442476622454,-0.3237449750307237 : 37.8667981224592296,0.0442607654259021 // from QGIS

/* 
var bbox = 
      {
        "type": "rectangle"
      }
    ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
              // 37.4766284489175, 37.9742531977015, -0.414203576514559, 0.149703611164146
              // (xmin, xmax, ymin, ymax)
              [[[37.4766284489175, 0.149703611164146], 
              [37.4766284489175, -0.414203576514559],
              [37.9742531977015, -0.414203576514559],
              [37.9742531977015, 0.149703611164146]]], null, false), 
            {
              "system:index": "0"
            })]);
*/
 
// var bbox = 
//     /* color: #d63000 */
//     /* displayProperties: [
//       {
//         "type": "rectangle"
//       }
//     ] */
//     ee.Geometry.Polygon(
//         [[[37.50, 0.20],
//           [37.50, -0.45],
//           [37.99, -0.45],
//           [37.99, 0.20]]], null, false);

/*
// Add to map 
Map.addLayer(bbox, {color: 'blue'}, 'bbox Kenya study area', true, 0.5);

// Compute the Normalized Difference Vegetation Index (NDVI).
// ((NIR - RED) / (NIR + RED + L))
var red = Landsat.select('SR_B4');
var nir = Landsat.select('SR_B5');
var ndvi = nir.subtract(red).divide(nir.add(red)).rename('ndvi');

// Display the result
var ndviParams = {min: -1, max: 1, palette: ['blue', 'white', 'green']};
Map.addLayer(ndvi, ndviParams, 'NDVI', false);

// Compute the Soil Adjusted Vegetation Index (SAVI)
// In Landsat 8-9:  
// SAVI = ((Band 5 – Band 4) / (Band 5 + Band 4 + 0.5)) * (1.5). :: https://www.usgs.gov/landsat-missions/landsat-soil-adjusted-vegetation-index

// Compute the Soil Adjusted Vegetation Index (SAVI)
var savi = Landsat.expression(
    '1.5*((NIR-RED)/(NIR+RED+0.5))',{
        'NIR':Landsat.select('SR_B4'),
        'RED':Landsat.select('SR_B5')
    }).rename('savi');

var saviVis = {'min':-1, 'max':1, 'palette':['red', 'yellow', 'green']};
Map.addLayer(savi, saviVis, 'SAVI', false);
*/

/////////////////////////////////////////////////////////////////////////////////
/// Spectral indexes
/////////////////////////////////////////////////////////////////////////////////

/*

ARVI    | Atmospherically Resistant Vegetation Index              | Vegetation in adverse conditions (indirect carbon).     | Vegetation                      |
EVI     | Enhanced Vegetation Index                               | Vegetation and indirect carbon.                         | Vegetation                      |
GNDVI   | Green Normalized Difference Vegetation Index            | Vegetation (chlorophyll) and indirect carbon.           | Vegetation                      |
KNDVI   | Kernel Normalized Difference Vegetation Index           | Vegetation in complex scenarios (indirect carbon).      | Vegetation                      |
MNDVI   | Modified Normalized Difference Vegetation Index         | Dense vegetation and indirect carbon (more sensitive).  | Vegetation                      |
MSAVI   | Modified Soil-Adjusted Vegetation Index                 | Vegetation with less soil noise (indirect carbon).      | Vegetation                      |
NDVI    | Normalized Difference Vegetation Index                  | Vegetation and indirect carbon (broad base).            | Vegetation                      |
NIRv
RVI (sent 2)| Ratio Vegetation Index                                    | Plant biomass and indirect carbon.                      | Vegetation                      |
SAVI    | Soil-Adjusted Vegetation Index                          | Vegetation adjusted for soil (indirect carbon).         | Vegetation                      |
BI      | Brightness Index                                        | Overall soil reflectance.                               | Soil Properties: Reflectance    |
DBSI    | Bare Soil Index                                         | Identification of exposed soil (low organic matter).    | Soil Properties: Exposure       |
NSDS    | Normalized Shortwave Infrared Difference Soil-Moisture  | Soil properties and carbon retention.                   | Soil Properties: Composition    |
RI      | Redness Index                                           | Coloration and organic matter.                          | Soil Properties: Color          |
SEVI    | Shadow-Eliminated Vegetation Index 
BAI     | Burned Area Index                                       | Detection of burned areas (carbon impact).              | Burned Areas                    |
CSI     | Char Soil Index                                         | Soils affected by burns (carbon).                       | Burned Areas                    |
NBR2     | Normalized Burn Ratio                                   | Burns and their effects on soil/vegetation (carbon).    | Burned Areas                   | NBR2 (Band 11 - Band 12)/(Band 11 + Band 12)  Dvorakova et al. (2020) 
NDMI    | Normalized Difference Moisture Index                    | Vegetation and soil moisture (carbon dynamics).         | Moisture                        |


// From https://doi.org/10.1016/j.rse.2018.09.015 
TVI: Transformed Vegetation Index.
MSI: Moisture Stress Index.
GRVI: Green Ratio Vegetation Index.
LSWI: Land Surface Water Index.
TSAVI: Transformed Soil-Adjusted Vegetation Index.
WDVI: Weighted Difference Vegetation Index.	

*/


// Indexes list 
var indices = [
               'ARVI', 'EVI', 'GNDVI', 'kNDVI', 'MNDVI', 'MSAVI', 'NDVI', 'NIRv', // 'RVI', // sent 2
               'SAVI', 'BI', 'DBSI', 'NSDS', 'RI', 'SEVI', 'BAI', 'CSI', 'NBR2', 'NDMI', 
               'TVI', 'MSI', 'GRVI', 'LSWI', 'TSAVI', 'ATSAVI', 'WDVI'
               ];

/*
========
SPECTRAL
========

-----------------------------------------
Example 6: Spectral Indices for Landsat-9
-----------------------------------------

=================
GitHub Repository
=================

Awesome spectral: https://github.com/davemlz/awesome-spectral-indices
Awesome Spectral Indices module for GEE: https://github.com/davemlz/spectral
https://github.com/awesome-spectral-indices/spectral?tab=readme-ov-file
https://awesome-ee-spectral-indices.readthedocs.io/en/latest/list.html

*/

// REQUIRE THE SPECTRAL MODULE
var spectral = require("users/dmlmont/spectral:spectral");

// REQUIRE THE PALETTES MODULE
var palettes = require('users/gena/packages:palettes');

// VIRIDIS PALETTE
var viridis = palettes.matplotlib.viridis[7];

// LOCATION OF INTEREST
// var palomino = ee.Geometry.Point([-73.5549,11.2503]);
var bbox = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.FeatureCollection(
        [ee.Feature(
            ee.Geometry.Polygon(
              // 37.4766284489175, 37.9742531977015, -0.414203576514559, 0.149703611164146
              // (xmin, xmax, ymin, ymax)
              [[[37.4766284489175, 0.149703611164146], 
              [37.4766284489175, -0.414203576514559],
              [37.9742531977015, -0.414203576514559],
              [37.9742531977015, 0.149703611164146]]], null, false), 
            {
              "system:index": "0"
            })]);
 

// FILTER THE DATASET
var L9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2') // get the image collection 
    // .filterDate('2014-01-01', '2023-12-31')  // Adjust the year interval (done below in calendar range)
    .filterBounds(bbox)
    .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']) 
    .filter(ee.Filter.calendarRange(1,2,'month')) // Dry season: 12-1; 7-9 :: https://gis.stackexchange.com/a/256334/178680
    .filter(ee.Filter.lte('CLOUD_COVER', 0.1)).median();  // Filter cloud by percent 
    //.sort('CLOUD_COVER').first();
// var L9 = Landsat.filter(ee.Filter.calendarRange(2014,2023,'year'))
//     .filter(ee.Filter.calendarRange(1,2,'month')); // Dry season filter :: https://gis.stackexchange.com/a/256334/178680
  
// SCALE THE IMAGE (LANDSAT 9 PARAMETERS ARE THE SAME AS LANDSAT 8)
var L9 = spectral.scale(L9,'LANDSAT/LC08/C02/T1_L2');
var L9 = spectral.offset(L9,'LANDSAT/LC08/C02/T1_L2');

// Landsat
var N = L9.select("SR_B5"); 
var R = L9.select("SR_B4"); 
var B = L9.select("SR_B2"); 
var G = L9.select("SR_B3"); 
var S1 = L9.select("SR_B6"); 
var S2 = L9.select("SR_B7"); 

// KERNEL TO USE
var kernel = "RBF";

// REQUIRED PARAMETERS ACCORDING TO THE REQUIRED BANDS
var parameters = {
  // Required bands for NDVI:
  "N": N, // Nir band
  "R": R, // Red band
  "B": B, // Blue band
  "G": G, // Green band
  "S2": S2, // SWIR2 band
  "S1": S1, // SWIR1 band
  "gamma": 1, // (default: 1) Weighting coefficient used for ARVI
  "g": 2,     // (default: 2.5) Gain factor for EVI, use 2 to compensate L = 0.5
  "C1": 6.5,  // (default: 6) Coefficient 1 for the aerosol resistance term for EVI, use 6.5 to compensate L = 0.5 (adjust the Red band contribution in the denominator, mitigating atmphosferic dispersion)
  "C2": 8,    // (default: 7.5) Coefficient 2 for the aerosol resistance term EVI, use 8 to compensate L = 0.5 (adjust the Blue band contribution, related atmphosferic dispersion, sensible to aerosols)
  "L": 0.5, // Default Canopy Background for SAVI and EVI, dense vegetaion lower L, sparse veg higher L
  "fdelta": 0.581, // (default: 0.581) Adjustment factor used for SEVI
  "sla": 1, // (default: 1) Soil line slope for TSAVI
  "slb": 0, // (default: 1) Soil line intercept for TSAVI
  
  // Required bands for kNDVI: k(N,N) and k(N,R)
  // When using the RBF kernel, k(N,N) = 1.0
  "kNN": 1.0,
  "kNR": spectral.computeKernel(L9, kernel, {
    "a": N,
    "b": R,
    "sigma": N.add(R).divide(2),
  })
};

print('Paramenters indexes', parameters);

// COMPUTE THE NIRv
// var L9 = spectral.computeIndex(L9,["NDVI","SAVI", "NIRv"], parameters);
var L9 = spectral.computeIndex(L9, indices, parameters);

// CHECK THE NEW BANDS to L9
print("New bands added to L9", L9.bandNames());

// ADD THE NEW BAND TO THE MAP
Map.addLayer(L9,{"min":0,"max":0.3,"bands":["SR_B4","SR_B3","SR_B2"]},"RGB");
Map.centerObject(bbox, 8); // zoom

/*
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "ARVI" ,"palette":viridis},"ARVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "EVI" ,"palette":viridis},"EVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "GNDVI" ,"palette":viridis},"GNDVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "kNDVI" ,"palette":viridis},"kNDVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "MNDVI" ,"palette":viridis},"MNDVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "MSAVI" ,"palette":viridis},"MSAVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "NDVI" ,"palette":viridis},"NDVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "NIRv" ,"palette":viridis},"NIRv");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "SAVI" ,"palette":viridis},"SAVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "BI" ,"palette":viridis},"BI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "DBSI" ,"palette":viridis},"DBSI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "NSDS" ,"palette":viridis},"NSDS");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "RI" ,"palette":viridis},"RI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "SEVI" ,"palette":viridis},"SEVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "BAI" ,"palette":viridis},"BAI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "CSI" ,"palette":viridis},"CSI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "NBR2" ,"palette":viridis},"NBR2");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "NDMI" ,"palette":viridis},"NDMI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "TVI" ,"palette":viridis},"TVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "MSI" ,"palette":viridis},"MSI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "GRVI" ,"palette":viridis},"GRVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "LSWI" ,"palette":viridis},"LSWI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "TSAVI" ,"palette":viridis},"TSAVI");
Map.addLayer(L9,{"min":0,"max":0.5,"bands": "ATSAVI" ,"palette":viridis},"ATSAVI");
*/

Map.addLayer(L9.clip(bbox),{"min":0,"max":0.5,"bands": "WDVI" ,"palette":viridis},"WDVI");



/////////////////////////////////////////////////////////////////////////////////
/// Geological indexes

/*
https://doi.org/10.1007/s11600-022-00814-7 Geological mapping using extreme gradient boosting and the deep neural networks

Landsat ratios 5/4*3/4 were used, since they can be used to distinguish between volcanic and metamorphic rocks from sedimentary rocks (Kusky and Ramadan 2002); 
ratios (4/2, 6/7, 6/5 in RGB), called Sabin’s ratios (Sabins 1999), were used to map iron oxides, clay minerals and ferrous minerals (Ourhzif et al. 2019).
*/

// Compute ratio 543
var ratio543 = L9.expression(
    '(b5/b4)*(b3/b4)',{
        'b3':G,
        'b4':R,
        'b5':N
    }).rename('ratio543');

// Compute ratio 543 and Sabin indexes
var Sab42 = R.divide(B).rename('Sab42');
var Sab67 = S1.divide(S2).rename('Sab67');
var Sab65 = S1.divide(N).rename('Sab65');


///////////////////////////////////////////// 
// Stack images to one assets 
var predictors = dem//.addBands(slope) // coming from TAGEE
                    //.addBands(aspect) // coming from TAGEE
                    //.addBands(hillshade) // droped beacuse hillshade is a visualization composed by aspect and slope 
                    //.addBands(northernness) // coming from TAGEE
                    .addBands(tpi)
                    .addBands(chili)
                    .addBands(landform)
                    .addBands(topo_diver)
                    .addBands(flow_dir)
                    .addBands(flow_accumul)
                    //.addBands(river_chann_width) // for Mount Kenya is not relavant
                    .addBands(landcover)
                    .addBands(terr_attrib)
                    .addBands(temp_avg)
                    .addBands(preciptation)
                    .addBands(L9)
                    // .addBands(ndvi) // replace by spectral indexes
                    // .addBands(savi) // replace by spectral indexes
                    .addBands(ratio543)
                    .addBands(Sab42)
                    .addBands(Sab67)
                    .addBands(Sab65)
                    ;

// clip for shapefile
var predictors_bbox = predictors.clip(bbox);
print('Check Predictors_bbox', predictors_bbox);


///////// Error: Exported bands must have compatible data types; found inconsistent types: Float32 and Byte
// Solution convert all to Float (to same datatype)  
predictors_bbox = predictors_bbox.toFloat(); 

//// Ai info (to check)
// float: save more memory less precision compared with double, but still huge number, can get 4 bytes of memory  
// double: use in high precision calculations like scientific calculatiosn, can get 8 bytes of memory  


/////////////////////////////////////////////////////////////////////////////////////////
/// Export a Cloud Optimized GeoTIFF (COG) by setting the "cloudOptimized"
/////////////////////////////////////////////////////////////////////////////////////////
// parameter to true
Export.image.toDrive({
 image: predictors_bbox,
 description: 'Covariates_Kenya',
 folder: 'earth_engine_Kenya',
 region: bbox,
 scale: 30,
 crs: 'EPSG:4326',
 formatOptions: {
   cloudOptimized: true
 }
});

/*
// Set the export "scale" and "crs" parameters.
Export.image.toAsset({
  // image: predictors,
  image: predictors_bbox,
  description: 'image_export',
  assetId: 'users/gelsleichter/all_predictors_bbox_epsg_4326_buff_2km', 
  region: bbox,
  scale: 30,
  crs: 'EPSG:4326',
  maxPixels: 1e13
  // crs: dem_crs       // Error: Projection: The CRS of a map projection could not be parsed. (Error code: 3)
  // scale: dem_scale   // does not catch, then you have to state as 30
});
*/



/*
End script
*/




/////////////////////////////////////////////////////////////////////////////////////////
/// Add each band as a layer
/////////////////////////////////////////////////////////////////////////////////////////

// Function to add each band as a layer to the map
function addBandsToMap(image, bandNames) {
  bandNames.evaluate(function(bandList) {
    bandList.forEach(function(band) {
      var bandImage = image.select(band);
      Map.addLayer(bandImage, {}, band);
    });
  });
}

// Add each band from predictors to the map
// addBandsToMap(predictors_bbox, predictors_bbox.bandNames());