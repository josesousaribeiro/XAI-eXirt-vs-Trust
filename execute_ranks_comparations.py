import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def bumpchart(df, show_rank_axis= True, rank_axis_distance= 1.1, 
              ax= None, scatter= False, holes= False,
              line_args= {}, scatter_args= {}, hole_args= {}):
    
    if ax is None:
        left_yaxis= plt.gca()
    else:
        left_yaxis = ax

    # Creating the right axis.
    right_yaxis = left_yaxis.twinx()
    
    axes = [left_yaxis, right_yaxis]
    
    # Creating the far right axis if show_rank_axis is True
    if show_rank_axis:
        far_right_yaxis = left_yaxis.twinx()
        axes.append(far_right_yaxis)
    
    for col in df.columns:
        y = df[col]
        x = df.index.values
        # Plotting blank points on the right axis/axes 
        # so that they line up with the left axis.
        for axis in axes[1:]:
            axis.plot(x, y, alpha= 0)

        left_yaxis.plot(x, y, **line_args, solid_capstyle='round')
        
        # Adding scatter plots
        if scatter:
            left_yaxis.scatter(x, y, **scatter_args)
            
            #Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color= bg_color, **hole_args)

    # Number of lines
    lines = len(df.columns)

    y_ticks = [*range(1, lines + 1)]
    
    # Configuring the axes so that they line up well.
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.5, 0.1))
    
    # Sorting the labels to match the ranks.
    left_labels = df.iloc[0].sort_values().index
    right_labels = df.iloc[-1].sort_values().index
    
    left_yaxis.set_yticklabels(left_labels)
    right_yaxis.set_yticklabels(right_labels)
    
    # Setting the position of the far right axis so that it doesn't overlap with the right axis
    if show_rank_axis:
        far_right_yaxis.spines["right"].set_position(("axes", rank_axis_distance))
    
    return axes

def retornRankPositions(df):
    ref_sequence = list(df.iloc[:,0]) #primeira coluna sempre sera a referencia
    print(ref_sequence)
    df_tmp = df.copy()
    for col in df.columns:
        for idx1, val1 in enumerate(ref_sequence):
            for idx2, val2 in enumerate(df[col]):
                if val1 == val2:
                    df_tmp.at[idx1,col] = str(idx2) +'_'+ val2
    return df_tmp
             

def plotBumpChart(df_features_rank_copy):
    

    #data = {"A":[1,2,1,3],"B":[2,1,3,2],"C":[3,3,2,1]}
    #df = pd.DataFrame(data, index=['step_1','step_2','step_3','step_4'])

    plt.figure(figsize=(15, 10))
    
    df_transformed = retornRankPositions(df_features_rank_copy)
    
    bumpchart(df_transformed.transpose(), show_rank_axis= True, scatter= True, holes= False,
            line_args= {"linewidth": 3, "alpha": 0.5}, scatter_args= {"s": 1, "alpha": 0.8}) ## bump chart class with nice examples can be found on github
    plt.show()

    colors = ['#1f77b4',
              '#ff7f0e',
              '#ec96aa',
              '#aec7e8',
              '#ffbb78',
              '#2ca02c',
              '#98df8a',
              '#ff9896',
              '#9467bd',
              '#c5b0d5',
              '#9cb4b7',
              '#c49c94',
              '#e377c2',
              '#f7b6d2',
              '#7f7f7f',
              '#c7c7c7',
              '#bcbd22',
              '#dbdb8d',
              '#17becf',
              '#9edae5',
              '#1f77b4',
              '#ff7f0e',
              '#ec96aa',
              '#aec7e8',
              '#ffbb78',
              '#2ca02c',
              '#98df8a',
              '#ff9896',
              '#9467bd',
              '#c5b0d5',
              '#9cb4b7',
              '#c49c94',
              '#e377c2',
              '#f7b6d2',
              '#7f7f7f',
              '#c7c7c7',
              '#bcbd22',
              '#dbdb8d',
              '#17becf',
              '#9edae5']

    

    plt.savefig('ranks_comparations.png')


df = pd.read_csv('df_explanation_analysis.csv',sep=',',index_col=0)

print(df)
df = df[['shap_mlp_x_test_original','eXirt_mlp_x_test_original', 'skater_mlp_x_test_original', 'eli5_mlp_x_test_original', 'dalex_mlp_x_test_original']]
plotBumpChart(df)