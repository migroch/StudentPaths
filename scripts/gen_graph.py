import numpy as np
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import json
import re
#import pdb
pd.set_option('mode.chained_assignment', None)


### Definitions ###
node_attributes = {
    'norecordfound': {'long_name': 'No College Record Found', 'sort_order': 0, 'color': '#7f7f7f' }, 
    'hsgrad': {'long_name': 'Graduated | High School', 'sort_order': 1, 'color': '#7D5BA6'},
    'enr<2years': {'long_name': 'Enrolled | Less Than 2 Years', 'sort_order': 2, 'color': '#bcbd22'},  
    'grad<2years': {'long_name': 'Graduated | Less Than 2 Years', 'sort_order': 3, 'color': '#387780'},
    'enr2year': {'long_name': 'Enrolled | 2-Year College', 'sort_order': 4, 'color': '#d62728'},
    'enr2year,enr<2years': {'long_name': 'Enrolled | 2-Year & Less Than 2 Years', 'sort_order': 5, 'color': '#ff9896'},
    'enr2year,enr4year': {'long_name': 'Enrolled | 2-Year & 4-Year ', 'sort_order': 6, 'color': '#55D6BE'},
    'grad2year': {'long_name': 'Graduated | 2-Year College', 'sort_order': 7, 'color': '#2ca02c'},
    'enr4year,enr<2years': {'long_name': 'Enrolled | 4-Year & Less Than 2 Years', 'sort_order': 8, 'color': '#9467bd'},
    'enr4year': {'long_name': 'Enrolled | 4-Year College', 'sort_order': 9, 'color': '#1f77b4'},
    'grad4year': {'long_name': 'Graduated | 4-Year College', 'sort_order': 10, 'color': '#ff7f0e'},
}

def read_nsch_data(path, max_year=None):
    '''
    Read NSCH data from specified path
    '''
    nsch_data = pd.read_csv('data/scz_county_ns_clearinghouse_detail.csv')
    # Filter
    nsch_data['HIGH_SCHOOL_GRAD_DATE'] = pd.to_datetime(nsch_data['HIGH_SCHOOL_GRAD_DATE'])
    if max_year:
        nsch_data = nsch_data[nsch_data.HIGH_SCHOOL_GRAD_DATE.dt.year < max_year]  
    return nsch_data

def gen_nodes_df(nsch_data):
    '''
    Defines raw_nodes from the NSCH data and assigns unique student IDs to nodes
    '''
    graph_time_unit = 4 # 1-year = 3 months*4 (Assuming a 3-month time unit in the nsch_data)

    # Define raw_nodes as (GradEnrollGroups, GRAPH_TIME) combinations
    nodes_df = nsch_data[['ID', 'GradEnrollGroups', 'TIME_FROM_HSGRAD']]  
    nodes_df['GRAPH_TIME'] = nodes_df['TIME_FROM_HSGRAD']/graph_time_unit
    nodes_df['GRAPH_TIME'] = nodes_df['GRAPH_TIME'].apply(np.ceil).astype('Int64')
    nodes_df['GRAPH_TIME'][nodes_df['GRAPH_TIME']==0] = 1 # GRAPH_TIME=0 for negative values, students enrolling before hs grad. Set those to 1
    nodes_df['GRAPH_TIME'][nodes_df['GRAPH_TIME'].isna()] = 1
    nodes_df = nodes_df.sort_values('GRAPH_TIME')

    # Set node IDs
    nodes_df['nodeID'] = (
        nodes_df.GradEnrollGroups
        .str.replace(' ','')
        .str.replace('|','')
        .str.replace('-','')
        .str.replace('College','')
        .str.replace('Enrolled','enr')
        .str.replace('Graduated','grad')
        .str.replace('LessThan','<')
        .str.lower()
    )  + '-' + nodes_df.GRAPH_TIME.astype('str')

    nodes_df = nodes_df.groupby(['ID','nodeID']).head(1) # selects unique (ID, nodeID) combinations
    return nodes_df

def gen_edges_df(nodes_df, time_max=None):
    '''
    Generate the graph edges from the graph data frame
    '''
    if not time_max: time_max = nodes_df.GRAPH_TIME.max()

    # Generate edges across time for each ID
    edges = pd.DataFrame([] , columns=['ID','Source', 'Target'])
    for ID in nodes_df['ID'].unique():
        thisIDnodes = nodes_df.query('ID==@ID')[['GRAPH_TIME', 'nodeID']]
        for time in range(0, time_max):
            source = thisIDnodes[thisIDnodes['GRAPH_TIME'] == time]['nodeID'].unique()
            target =  thisIDnodes[thisIDnodes['GRAPH_TIME'] == time+1]['nodeID'].unique()
            source = reduce_nodes(source)
            target = reduce_nodes(target)
            if time == 0:
                # add 0->1 edge to all
                source = 'hsgrad-0' # at t=0 the source node for all is HS graduation
            if not len(source):
                print(f'WARNING: no source node for ID:{ID} at time:{time}')
                #source = '-'+str(time)
            if not len(target):
                target = re.sub('-\d', '-'+str(time+1), source)
                thisIDnodes = thisIDnodes.append(pd.DataFrame([[time+1, target]], columns=['GRAPH_TIME', 'nodeID']))
            edges = edges.append(pd.DataFrame([[ID, source, target]], columns=['ID', 'Source', 'Target']))      
    return edges

def reduce_nodes(nodes):
    '''
    Return a single node from any nodes found at a given time for a student ID
    '''
    if len(nodes) == 1:
        node = nodes[0]
    elif len(nodes) > 1:
        nodes_grad = [s for s in nodes if 'grad' in s]
        nodes_enr = [s for s in nodes if 'enr' in s]
        if len(nodes_grad):
            node = nodes_grad[-1]
        else:
            nodes = nodes_enr    
            node = ','.join(np.sort(nodes))
    else:
        node = ''
    return node

def gen_graph(edges):
    '''
    Generate the Networkx Graph (G) from the edges data frame
    '''
    # Group edges by Source-Target        
    grp_edges = edges.groupby(['Source', 'Target']).agg(
        count = pd.NamedAgg(column='ID', aggfunc='nunique'),
        ids =  pd.NamedAgg(column='ID', aggfunc='unique')
    )
    grp_edges = grp_edges.reset_index()
    grp_edges['ids'] = grp_edges['ids'].apply(lambda x: x.tolist())

    # Generate Graph
    G = nx.from_pandas_edgelist(grp_edges, source='Source', target='Target',
                                edge_attr=['count', 'ids'], create_using=nx.DiGraph) 
        
    # Set node attributes
    node_attrs = {}
    for node in G.nodes:
        if len(node.split('-'))<3:
            node_key = node.split('-')[0]
        else:
            node_key = node.split('-')[0] + node.split('-')[1][1:]
        node_attrs[node] = {
            'name': node,
            'time': node.split('-')[-1],
            'long_name': node_attributes[node_key]['long_name'], 
            'sort_order': node_attributes[node_key]['sort_order'],
            'color': node_attributes[node_key]['color'],
        }
    nx.set_node_attributes(G, node_attrs)
    return G

def write_graph(G, path):    
    '''
    Write Graph in json format
    '''
    G_json = json_graph.node_link_data(G)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(G_json, f, ensure_ascii=False, indent=4) 


if __name__ ==  "__main__":
    print('Reading NSCH data')
    nsch_data = read_nsch_data('data/scz_county_ns_clearinghouse_detail.csv', max_year=2023)

    print('\nGenerating pre-graph df from NSCH data')
    nodes_df = gen_nodes_df(nsch_data)
    raw_nodes = nodes_df[['ID','nodeID']].groupby('nodeID')['ID'].unique()

    print('\nGenerating the edges data frame')
    edges = gen_edges_df(nodes_df)

    print('\nCreating Graph')
    G = gen_graph(edges)

    write_path = 'data/graph_timestep1yr.json'
    print(f'\nWriting graph to file {write_path}')
    write_graph(G, write_path)
