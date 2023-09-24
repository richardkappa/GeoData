import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import pickle
from sklearn.neighbors import BallTree
import pygeos
from libpysal import weights
import networkx as nx

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


#Density within a given radius
def average_within_radius(src_points, #Source gdf, the gdf with the points you want to find the average near
                          candidates, #gdf containing the list of points you want to find the average of
                          factor,     #factor in your candidates gdf you want the average of
                          radius      #radius you want to find the density inside of in metres
                          ):

    src = src_points.copy().reset_index(drop=True)
    can = candidates.copy().reset_index(drop=True)
    values = can.loc[:,factor].to_numpy()

    src_geom_col = src.geometry.name
    can_geom_col = can.geometry.name

    np_df = np.array(src[src_geom_col].apply(lambda geom: (geom.x, geom.y)).to_list())
    np_points = np.array(can[can_geom_col].apply(lambda geom: (geom.x, geom.y)).to_list())
    tree = BallTree(np_points, leaf_size=15)

    idx = tree.query_radius(np_df, r=radius, count_only=False, return_distance =False, sort_results=False)

    #find the average value weighted by the number of values
    avg = [np.nan if len(x)==0 else values[x].mean() for x in idx]

    return avg


#Generalised function to find the nearest candidate geometry to each source geometry
#returns a value of -1 if there are none
def dist_to_nearest(source, candidates, return_geom = False):
    source_py = pygeos.from_shapely(source.geometry)
    candidates_py = pygeos.from_shapely(candidates.geometry)
    tree = pygeos.STRtree(candidates_py)
    s_near, c_near = tree.nearest(source_py)

    dist = pygeos.distance(source_py[s_near.tolist()], candidates_py[c_near.tolist()])
    if len(dist)==0:
        dist = np.full(len(source), -1, dtype="int64")

    if return_geom:
        out = dist, candidates.loc[c_near.tolist() ,"geometry"]
    else:
        out = dist

    return out

#Generalised function to find number of geometries within a radius of a point in the source geometry
def within_radius(source, candidates, radius):
    source_py = pygeos.from_shapely(source.geometry)
    candidates_py = pygeos.from_shapely(candidates.geometry)

    tree = pygeos.STRtree(candidates_py)
    s_idx, c_idx = tree.query_bulk(source_py, predicate='dwithin', distance=radius)
    if len(np.bincount(s_idx))==0:
        print("No points in the radius")
        return np.full(len(source), 0, dtype="int64") 
    else:
        tail = np.full(len(source) - len(np.bincount(s_idx)), 0, dtype="int64")
        return np.concatenate((np.bincount(s_idx), tail), axis=0)

##Network analysis functions
#Converts a gepandas dataframe of polygones into a networkx network of boundaries
def create_network_from_shapes(shape_gdf):
    
    #Create a weight object from the gdf, this object contains all boundaires between polygons that are next to each other. The convert it to a networkx object
    queen = weights.Queen.from_dataframe(shape_gdf)
    G = queen.to_networkx()

    #Name the nodes in this network the same as the names of the polygon names
    keys = [*range(0,G.number_of_nodes())]
    values = shape_gdf["Name"].to_list()
    Node_names = dict(zip(keys, values))
    G = nx.relabel_nodes(G, Node_names)

    #For plotting on a chart, get the locations of hte polygon centroids for the locations of each node
    centroids = np.column_stack((shape_gdf.centroid.x, shape_gdf.centroid.y))
    positions = dict(zip(G.nodes, centroids))

    return G, positions

#returns a sub network of the nodes within netowork_size edges of your stated node
def witin_n_boundaries(G, node_name, network_size):
    G_n = nx.ego_graph(G, node_name, network_size)
    in_network = list(dict(G_n.nodes()).keys())

    return G_n, in_network

#Given a list of polygons in a given gdf, what is the average value in these polygons
def Average_In_Neighbourhood(shape_gdf, shape_list, count, average):
    new = shape_gdf[shape_gdf['Name'].isin(shape_list)].loc[:,[count, average]]
    sales = np.sum(new[count])
    print(sales)
    Av_Cost = 0
    # if sales == 0:
    #     Av_Cost = np.NaN
    # else:
    #     test = np.sum(new[count] * new[average])
    #     Av_Cost = np.round(np.divide(np.sum(new[count] * new[average]), sales))
    return [sales, Av_Cost]

#Combine the above two finctions into a single function we can use in a list comprehension in the next function
def average_within_n_boundaries_sub(shape_gdf, G, node_name, count, average, network_size):
    G_n, in_network = witin_n_boundaries(G, node_name, network_size=network_size)
    count_mean = Average_In_Neighbourhood(shape_gdf, in_network, count, average)
    return count_mean

#Combine all of the network functions above into a single function.
#For a given set of boundary shapefiles with average and count variables contained within. Find the average value of within network_size boundaries of each polygon
def average_within_n_boundaries(shape_gdf, count, average, network_size):
    G, pos = create_network_from_shapes(shape_gdf)

    count_name = count + "_within_" + str(network_size) + "_boundaries"
    average_name = average + "_within_" + str(network_size) + "_boundaries"
    count_mean = pd.DataFrame([average_within_n_boundaries_sub(shape_gdf, G, x, count, average, network_size) for x in shape_gdf["Name"]], columns = [count_name, average_name])

    return count_mean