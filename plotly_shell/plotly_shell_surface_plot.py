# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:53:12 2019

@author: Oliver B France
"""

import plotly as py
import plotly.graph_objs as go

import pandas as pd

# Read data from a csv
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')


df = pd.DataFrame(data=z_data)
df.head()

data = [
    go.Surface(
        z=z_data.as_matrix()
    )
]

layout = go.Layout(
    title='Mt Bruno Elevation',
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)
    
fig = go.Figure(data=data, layout=layout)

py.offline.plot(fig, filename='elevations-3d-surface.html')