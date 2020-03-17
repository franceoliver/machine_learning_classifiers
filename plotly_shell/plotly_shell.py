import pandas as pd 
import numpy as np 

from plotly import __version__
import cufflinks as cf
import plotly.offline 
import plotly
import plotly.graph_objs

init_notebook_mode(connected=True)

cf.go_offline

#DATA
df = pd.DataFrame(np.random.randn(100,4), columns='A B C D'.split())

df2 = pd.DataFrame({'Category':['A','B','C'], 'Values':[32,43,50]})

df3 = pd.DataFrame({'x':[1,2,3,4,5] , 'y':[10,20,30,40,50] , 'z':[500,400,300,200,100]})

# Plotting Graph (scatter)
plotly.offline.plot({'data':[plotly.graph_objs.Scatter(x=df['A'], # graph type and X axis
                                                       y=df['B'], # Y axis
                                                       )],
                     'layout':plotly.graph_objs.Layout(showlegend=True,
                                                       height=1000,
                                                       width=1200,
                                                       )
                     })

# Plotting Graph (scatter)
plotly.offline.plot({'data':[plotly.graph_objs.Scatter(x=df['A'], # graph type and X axis
                                                       y=df['B'], # Y axis
                                                       mode='markers',)],
                     'layout':plotly.graph_objs.Layout(showlegend=True,
                                                       height=1000,
                                                       width=1200,
                                                       )
                     })
                     

# Plotting Graph (Bar)
plotly.offline.plot({'data':[plotly.graph_objs.Bar(x=df2['Category'], # graph type and X axis
                                                       y=df2['Values'])],
                     'layout':plotly.graph_objs.Layout(showlegend=True,
                                                       height=1000,
                                                       width=1200,
                                                       )
                     })
                     

# Plotting Graph (Surface)
plotly.offline.plot({'data':[plotly.graph_objs.Surface(x=df3['x'], # graph type and X axis
                                                       y=df3['y'], # Y axis
                                                       z=df3['z'])],
                     'layout':plotly.graph_objs.Layout(showlegend=True,
                                                       autosize=True,
#                                                       height=1000,
#                                                       width=1200,
                                                       )
                     })
                     
# Plotting Graph (Histogram)
plotly.offline.plot({'data':[plotly.graph_objs.Histogram(x=df['A'], # graph type and X axis
                                                       )],
                     'layout':plotly.graph_objs.Layout(showlegend=True,
                                                       height=1000,
                                                       width=1200,
                                                       )
                     })    
                     
# Plotting Graph (Bubble)
plotly.offline.plot({'data':[plotly.graph_objs.Bubble(x=df['A'], # graph type and X axis
                                                       y=df['B'], # Y axis
                                                       mode='markers')],
                     'layout':plotly.graph_objs.Layout(showlegend=True,
                                                       height=1000,
                                                       width=1200,
                                                       )
                     })