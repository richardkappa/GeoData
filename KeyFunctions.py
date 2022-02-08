import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import pickle
from sklearn.neighbors import BallTree

#Import a points csv into a dataframe
def Import_Points(infile, inputs, dtypes):
    #print(inputs["Type"])
    #Get the before and after columns names
    usecols = list(filter(None,list(inputs.values())[1:8]))
    col_names = list({key: value for key, value in inputs.items() if value in usecols}.keys())
    
    #Get the data types
    t_dtypes = {key: value for key, value in dtypes.items() if key in col_names}
    dtypes_in = {inputs[k]:v for k, v in t_dtypes.items()}
    dtype_out = {key: value for key, value in dtypes.items() if key in col_names}

    #Import the data
    df = pd.read_csv(infile, usecols=usecols, encoding = "ISO-8859-1", dtype=dtypes_in) 

    #Reformat ready for next step
    df.columns = col_names
    df = df.astype(dtype_out)

    df["Type"] = inputs["Type"]
    return df


#Check if extra factors are in a list, used in get_shapefile
def getextra(list):
    try:
        extra = list[2]
        extraexists = True
    except:
        extra = []
        extraexists = False

    return extra, extraexists

#import a shapefile and put it into our standard format
def get_shapefile(name, details, crs):
    extra, extraexists = getextra(details)
    gdf = gpd.read_file(details[0]).to_crs(crs).loc[:,[details[1], "geometry"]+extra]
    gdf = gdf.rename(columns={details[1]: "Name"})
    gdf["Type"] = name

    usecols = ["Type", "Name", "geometry"]

    if extraexists:
        array = gdf[extra[0]].to_numpy()
        if len(extra)>1:
            for e in extra[1:]:
                array = np.column_stack((array,gdf[e].to_numpy()))
        gdf["Details_Str"] = array.tolist()
        usecols = usecols + ["Details_Str"]

    gdf = gdf[usecols]

    return gdf

#make a polygon
def MakePolygon(centre, sq_sz, inp_crs):
    mnx = centre[0]-(sq_sz/2)
    mny = centre[1]-(sq_sz/2)
    mxx = centre[0]+(sq_sz/2)
    mxy = centre[1]+(sq_sz/2)
    polygon =Polygon([(mnx, mny), (mnx,mxy), (mxx,mxy), (mxx,mny), (mnx, mny)])
    poly_gdf = gpd.GeoDataFrame(geometry=[polygon], crs=inp_crs)
    return poly_gdf

#load a saved pickle file
def load_obj(root_path, name):
    with open(root_path + 'WorkingData/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#Density within a given radius
def density_within_radius(src_points, #Source gdf, the gdf with the points you want to find the density near
                          candidates, #gdf containing the list of points you want to find the density of
                          radius      #radius you want to find the density inside of in metres
                          ):
    df_geom_col = src_points.geometry.name
    points_geom_col = candidates.geometry.name

    np_df = np.array(src_points[df_geom_col].apply(lambda geom: (geom.x, geom.y)).to_list())
    np_points = np.array(candidates[points_geom_col].apply(lambda geom: (geom.x, geom.y)).to_list())
    tree = BallTree(np_points, leaf_size=15)

    within_radius = tree.query_radius(np_df, r=radius, count_only=True)
    return within_radius



def get_nearest_point(src_points, #Source gdf, the gdf with the points you want to attach distances to
                      candidates, #gdf containing the list of points you want to get the distances to
                      k_neighbors=1 #Return just the closest point
                     ):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
     # removed metric since the default is euclidian (what my coordinates use)
    tree = BallTree(candidates, leaf_size=15)

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)

def nearest_neighbor_point(left_gdf, #source data
                         right_gdf, #points data
                         right_col_name, #in the points data, what column contains the name of the points
                         outname, #in the output data what do you want to refer to the points as
                         keep_n_g_d = [True, True, True], #do you want to keep the geometry, point name and distance
                         merge=False #Make a new output or merge onto left_gdf
                        ):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    """
    
    suffix = "Nearest_" + outname
        
    if len(right_gdf)>0:

        # Ensure that index in right gdf is formed of sequential numbers
        left = left_gdf.copy().reset_index(drop=False)
        right = right_gdf.copy().reset_index(drop=True)
        
        left_geom_col = left.geometry.name
        right_geom_col = right.geometry.name
        
        #Some of the data frames will have empty geometries so this will drop them
        left = left[~(left.is_empty)]
        right = right[~(right.is_empty)]
        
        left = left.copy().reset_index(drop=False)
        right = right.copy().reset_index(drop=False)

        left_points = np.array(left[left_geom_col].apply(lambda geom: (geom.x, geom.y)).to_list())
        right_points = np.array(right[right_geom_col].apply(lambda geom: (geom.x, geom.y)).to_list())

        # Find the nearest points
        # -----------------------
        # closest ==> index in right_gdf that corresponds to the closest point
        # dist ==> distance between the nearest neighbors (in meters)

        closest, dist = get_nearest_point(src_points=left_points, candidates=right_points)

        # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
        closest_points = right.loc[closest]

        # Ensure that the index corresponds the one in left_gdf
        left_index = left['index']
        closest_points = closest_points.reset_index(drop=True)
        closest_points = closest_points.set_index(left_index)

        #Select the columns to keep in the output dataframe
        keep = [right_col_name, 'geometry']  
        closest_points = closest_points[keep]

        #Rename the columns to keep
        closest_points = closest_points.rename(columns={right_col_name: suffix + '_name'})
        closest_points = closest_points.rename(columns={'geometry': suffix + '_geometry'})
        closest_points[suffix + '_Distance'] = dist

        closest_points = closest_points.loc[:,keep_n_g_d]

    else:
        closest_points = left_gdf.iloc[:,0:1].copy()
        closest_points[suffix + '_name'] = ""
        closest_points[suffix + '_geometry'] = ""
        closest_points = closest_points.loc[:,[suffix + '_name',suffix + '_geometry']]
        closest_points[suffix + '_Distance'] = np.nan
        closest_points = closest_points.loc[:,keep_n_g_d]
        
        
    # either add the columns to left_gdf or make a new dataframe
    if merge:
        out = left_gdf.join(closest_points)
    else:
        out = closest_points
    return out