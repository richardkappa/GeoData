# GeoData

Explorations into processing geographic data

## Data Sources

The data has all been downloaded to my local PC but it can be found in the locations below. It is all UK data with an open government licence, or equivalent. I've found other shapefiles online but was unsure on their licence requirements so have not used them

+ National statistics Geoportal
    + National Statistics Postcode Lookup File [NSPL](https://geoportal.statistics.gov.uk/)
    + shape files for Counties, National Parks, Countries, Local Authorities, LSOA 2011
+ Land Registry Price Paid data [PP_data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
+ Ordnance Survey OpenData
    + [Boundary-Line](https://osdatahub.os.uk/downloads/open/BoundaryLine)
    + [OS Open Greenspace](https://osdatahub.os.uk/downloads/open/OpenGreenspace)
    + [OS Open Rivers](https://osdatahub.os.uk/downloads/open/OpenRivers)
    + [OS Open Roads](https://osdatahub.os.uk/downloads/open/OpenRoads)
    + [OS Open UDPRN](https://osdatahub.os.uk/downloads/open/OpenUPRN) (Not used)
    + [Strategi] (As at 2016 and now no longer on the OS website)
        + Used for coastline, railways, ferries...
+ National Public Transport Access Nodes [(NaPTAN)](https://data.gov.uk/dataset/ff93ffc1-6656-47d8-9155-85ea0b8f2251/national-public-transport-access-nodes-naptan)
+ [DTF Road Safety Data](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data)
+ [Fire data](https://www.gov.uk/government/statistics/fire-statistics-incident-level-datasets)
+ [NHS data](https://www.nhs.uk/about-us/nhs-website-datasets/) 
+ [Schools](https://get-information-schools.service.gov.uk/)
+ [Natural England Data](https://naturalengland-defra.opendata.arcgis.com/)
+ [Historic England Data](https://historicengland.org.uk/listing/the-list/data-downloads/)
    + Listed Buildings
    + Conservation Areas
+ [Met Office Historic Data](https://www.metoffice.gov.uk/research/climate/maps-and-data/data/haduk-grid/datasets)

+ Not yet used, but will hopefully add
	+ Ordnance Survey OpenData
		+ [OS Terrain50](https://osdatahub.os.uk/downloads/open/Terrain50)
    + [BGS Data](https://www.bgs.ac.uk/geological-data/opengeoscience/map-data-downloads/)
    + [2021 census data](https://www.nomisweb.co.uk/census/2021/bulk)
        +This will take some thought as there are lots of individual files 

	
	

## Main
+ 01_GetData.ipynb
    + Imports & Processes all raw data into a standard format
+ 02_CreatePostcodeFile.ipynb
    + From the raw gdf made in part 1 it creates a dataframe for all UK postcodes with lots of calculated variables based on our inputs
+ KeyFunctions.py
    + The module containing the functions used by 01 and 02 and other items


## Investigations
This folder contains some of the tests done to create the code in the main folder
+ Maps.ipynb
    + An investigative piece of code containing different map outputs