import numpy as np
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph
import json

pd.set_option('mode.chained_assignment', None)

nsch_data = pd.read_csv('data/scz_county_ns_clearinghouse_detail.csv')

# Filter
nsch_data['HIGH_SCHOOL_GRAD_DATE'] = pd.to_datetime(nsch_data['HIGH_SCHOOL_GRAD_DATE'])
#nsch_data = nsch_data[nsch_data.HIGH_SCHOOL_GRAD_DATE.dt.year < 2015]  

graph_time_unit = 4 # 1-year = 3 months*4 (Assuming a 3-month time unit in the nsch_data)

graph_df = nsch_data[['ID', 'GradEnrollGroups', 'TIME_FROM_HSGRAD']]  
graph_df['GRAPH_TIME'] = graph_df['TIME_FROM_HSGRAD']/graph_time_unit
graph_df['GRAPH_TIME'] = graph_df['GRAPH_TIME'].apply(np.ceil).astype('Int64')
graph_df['GRAPH_TIME'][graph_df['GRAPH_TIME']==0] = 1
graph_df['GRAPH_TIME'][graph_df['GRAPH_TIME'].isna()] = 1
graph_df = graph_df.sort_values('GRAPH_TIME')

graph_df['nodeID'] = (
    graph_df.GradEnrollGroups
    .str.replace(' ','')
    .str.replace('|','')
    .str.replace('-','')
    .str.replace('College','')
    .str.replace('Enrolled','enr')
    .str.replace('Graduated','grad')
    .str.replace('LessThan','<')
    .str.lower()
)  + '-' + graph_df.GRAPH_TIME.astype('str')

graph_df = graph_df.groupby(['ID','nodeID']).head(1) # selects unique (ID, nodeID) combinations
raw_nodes = graph_df[['ID','nodeID']].groupby('nodeID')['ID'].unique()

def clean_nodes(nodes):
    '''
    Return a single node from any nodes found at a given time for a student ID
    '''
    if len(nodes) == 1:
        node = nodes[0]
    elif len(nodes) > 1:
        nodes_grad = [s for s in nodes if 'grad' in s]
        nodes_enr = [s for s in nodes if 'enr' in s]
        if len(nodes_grad):
            nodes = nodes_grad
        else:
            nodes = nodes_enr    
        node = ','.join(np.sort(nodes))
    else:
        node = ''
    return node

time_max = graph_df.GRAPH_TIME.max()
#time_max = 5

# Generate edges across time for each ID
edges = pd.DataFrame([] , columns=['ID','Source', 'Target'])
for ID in graph_df['ID'].unique():
    thisIDnodes = graph_df.query('ID==@ID')[['GRAPH_TIME', 'nodeID']]
    for time in range(0, time_max):
        source = thisIDnodes[thisIDnodes['GRAPH_TIME'] == time]['nodeID'].unique()
        target =  thisIDnodes[thisIDnodes['GRAPH_TIME'] == time+1]['nodeID'].unique()
        source = clean_nodes(source)
        target = clean_nodes(target)
        if time == 0:
            # add 0->1 edge to all
            source = 'hsgrad-0' # at t=0 the source node for all is HS graduation
        if not len(source):
            source = 'norecordfound-'+str(time)
        if not len(target):
            target = 'norecordfound-'+str(time+1)
        edges = edges.append(pd.DataFrame([[ID, source, target]], columns=['ID', 'Source', 'Target']))      

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
node_longNames = {
    'hsgrad': 'Graduated | High School',
    'enr2year': 'Enrolled | 2-Year College',
    'enr4year': 'Enrolled | 4-Year College',
    'enr2year,enr4year': 'Enrolled | 2-Year & 4-Year ',
    'enr<2years': 'Enrolled | Less Than 2 Years', 
    'grad2year': 'Graduated | 2-Year College',
    'grad4year': 'Graduated | 4-Year College',
    'grad2year,grad4year': 'Graduated | 2-Year & 4-Year ',
    'grad<2years': 'Graduated | Less Than 2 Years',
    'norecordfound': 'No College Record Found', 
}


node_attrs = {}
for node in G.nodes:
    if len(node.split('-'))<3:
        node_key = node.split('-')[0]
    else:
        node_key = node.split('-')[0] + node.split('-')[1][1:]
    node_attrs[node] = {
        'longName':  node_longNames[node_key], 
        'time': node.split('-')[-1]
    }
nx.set_node_attributes(G, node_attrs)

# Write Graph in json format
G_json = json_graph.node_link_data(G)
with open('data/graph_timestep1yr.json', 'w', encoding='utf-8') as f:
    json.dump(G_json, f, ensure_ascii=False, indent=4) 
