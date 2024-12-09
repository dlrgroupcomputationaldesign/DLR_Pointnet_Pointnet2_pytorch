import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import ast


full = True
_dir = "room_outputs"

#####################################################################################################################
"""
Either plots the full building in a plotly visual (full = True) or captures room plots from the building in the _dir
(if full = false).
"""

df_full = pd.read_csv(r"D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\visualizer\output.csv")
rooms = []
for index, row in df_full.iterrows():
    room_row = row['Room']
    room_row = ast.literal_eval(room_row)
    for room in room_row:
        rooms.append(room)

rooms_list = pd.Series(rooms).unique()
df = df_full.copy()
# Convert color and label columns to integers if necessary
df[['r', 'g', 'b', 'l']] = df[['r', 'g', 'b', 'l']].astype(int)  # Changed 'label' to 'l'

# Label mapping dictionary
label_mapping = {'Other': 0, 'Window': 1, 'Door': 2, 'Floor': 3, 'Roof': 4, 'Wall': 5, 'Ceiling': 6}

# Invert the label mapping dictionary to map numerical labels to string labels
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Map the numerical labels to string labels
df['Label_Name'] = df['l'].map(inverse_label_mapping)

# Define a color map for each label
color_map = {
    'Other': 'red',
    'Window': 'blue',
    'Door': 'green',
    'Floor': 'purple',
    'Roof': 'orange',
    'Wall': 'pink',
    'Ceiling': 'brown'
}

# Visualize using Plotly
fig = px.scatter_3d(df, x='x', y='y', z='z', color='Label_Name',
                    color_discrete_map=color_map,
                    labels={'Label_Name': 'Label'},
                    title='3D Point Cloud by Labels')

fig.update_traces(marker=dict(size=1))  # Increase the size of the markers in the plot

# Update the legend marker size
fig.update_layout(
    legend=dict(
        itemsizing='constant',
        itemwidth=30,
        tracegroupgap=0,
        font=dict(size=12)
    )
)

if full == True:
    fig.show()

else:
    for room in rooms_list:

        print(f"Outputing {room}")
        df = df_full[df_full['Room'].apply(lambda rooms: room in rooms)].copy()

        # Convert color and label columns to integers if necessary
        df[['r', 'g', 'b', 'l']] = df[['r', 'g', 'b', 'l']].astype(int)  # Changed 'label' to 'l'

        # Label mapping dictionary
        label_mapping = {'Other': 0, 'Window': 1, 'Door': 2, 'Floor': 3, 'Roof': 4, 'Wall': 5, 'Ceiling': 6}

        # Invert the label mapping dictionary to map numerical labels to string labels
        inverse_label_mapping = {v: k for k, v in label_mapping.items()}

        # Map the numerical labels to string labels
        df['Label_Name'] = df['l'].map(inverse_label_mapping)

        # Define a color map for each label
        color_map = {
            'Other': 'red',
            'Window': 'blue',
            'Door': 'green',
            'Floor': 'purple',
            'Roof': 'orange',
            'Wall': 'pink',
            'Ceiling': 'brown'
        }

        # Visualize using Plotly
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='Label_Name',
                            color_discrete_map=color_map,
                            labels={'Label_Name': 'Label'},
                            title='3D Point Cloud by Labels')

        fig.update_traces(marker=dict(size=1))  # Increase the size of the markers in the plot

        # Update the legend marker size
        fig.update_layout(
            legend=dict(
                itemsizing='constant',
                itemwidth=30,
                tracegroupgap=0,
                font=dict(size=12)
            )
        )
        # Save the figure
        try:
            pio.write_image(fig, f'{_dir}/{room}.png')
        except ValueError:
            print(f"Failed to write {room}")