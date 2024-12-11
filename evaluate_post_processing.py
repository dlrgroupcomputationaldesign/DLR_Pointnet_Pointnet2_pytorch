import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import ast
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from processing import postprocess

plot_floor_gram = False
full = True
_dir = "room_outputs"

#####################################################################################################################
"""
Either plots the full building in a plotly visual (full = True) or captures room plots from the building in the _dir
(if full = false).
"""





if __name__ == '__main__':
    df_full = pd.read_csv(r"D:\Repos\pointnetpytorch\DLR_Pointnet_Pointnet2_pytorch\visualizer\output1.csv")
    df = postprocess(df_full)

    # Visualize using Plotly
    fig = px.scatter_3d(df[(df['Label_Name'] == 'Wall') & (df['wall_label'] != -1)], x='x', y='y', z='z',
                        color='wall_label',
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
#
# else:
#     for room in rooms_list:
#
#         print(f"Outputing {room}")
#         df = df_full[df_full['Room'].apply(lambda rooms: room in rooms)].copy()
#
#         # Convert color and label columns to integers if necessary
#         df[['r', 'g', 'b', 'l']] = df[['r', 'g', 'b', 'l']].astype(int)  # Changed 'label' to 'l'
#
#         # Label mapping dictionary
#         label_mapping = {'Other': 0, 'Window': 1, 'Door': 2, 'Floor': 3, 'Roof': 4, 'Wall': 5, 'Ceiling': 6}
#
#         # Invert the label mapping dictionary to map numerical labels to string labels
#         inverse_label_mapping = {v: k for k, v in label_mapping.items()}
#
#         # Map the numerical labels to string labels
#         df['Label_Name'] = df['l'].map(inverse_label_mapping)

        # # Define a color map for each label
        # color_map = {
        #     'Other': 'red',
        #     'Window': 'blue',
        #     'Door': 'green',
        #     'Floor': 'purple',
        #     'Roof': 'orange',
        #     'Wall': 'pink',
        #     'Ceiling': 'brown'
        # }
        #
        # # Visualize using Plotly
        # fig = px.scatter_3d(df, x='x', y='y', z='z', color='Label_Name',
        #                     color_discrete_map=color_map,
        #                     labels={'Label_Name': 'Label'},
        #                     title='3D Point Cloud by Labels')
        #
        # fig.update_traces(marker=dict(size=1))  # Increase the size of the markers in the plot
        #
        # # Update the legend marker size
        # fig.update_layout(
        #     legend=dict(
        #         itemsizing='constant',
        #         itemwidth=30,
        #         tracegroupgap=0,
        #         font=dict(size=12)
        #     )
        # )
        # # Save the figure
        # try:
        #     pio.write_image(fig, f'{_dir}/{room}.png')
        # except ValueError:
        #     print(f"Failed to write {room}")

        # Define bin edges for x and y
        # get wals


