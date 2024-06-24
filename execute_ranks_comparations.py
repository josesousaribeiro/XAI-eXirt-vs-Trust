import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import util

from scipy import stats

bar = util.bar_system()
def bumpchart(df, show_rank_axis= True, rank_axis_distance= 1.1, 
              ax= None, scatter= False, holes= False,
              line_args= {}, scatter_args= {}, hole_args= {},
              color_dic=None):
    
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

        if color_dic != None:
            left_yaxis.plot(x, y, **line_args, color=color_dic[col], solid_capstyle='round')
        else:
            left_yaxis.plot(x, y, **line_args, solid_capstyle='round')    
        # Adding scatter plots
        
        if scatter:

            if color_dic != None:
                left_yaxis.scatter(x, y, color = color_dic[col], **scatter_args)
            else:
                left_yaxis.scatter(x, y,**scatter_args)
            #Adding see-through holes
            if holes:
                bg_color = left_yaxis.get_facecolor()
                left_yaxis.scatter(x, y, color = bg_color, **hole_args)


    # Number of lines
    lines = len(df.columns)

    y_ticks = [*range(0, lines)]
    
    # Configuring the axes so that they line up well.
    for axis in axes:
        axis.invert_yaxis()
        axis.set_yticks(y_ticks)
        axis.set_ylim((lines + 0.2, -1))
    
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
                    df_tmp.at[idx1,col] = idx2
    df_tmp['index'] = ref_sequence
    df_tmp = df_tmp.set_index('index')
    return df_tmp
             

def plotBumpChart(df_features_rank_copy,model,test,color_dic=None,fig_size=(3,2.5)):

    plt.figure(figsize=fig_size,dpi=300)
    plt.xticks(fontsize=9,rotation=90)
    plt.tight_layout(pad=4)
    plt.ylabel(test+'\n Relevance Rank',fontsize=9)
    sum_corr = 0
    for i,c in enumerate(df_features_rank_copy.columns):
        original = df_features_rank_copy.iloc[:,0] #ranque original
        if i > 0:
            rank = list(df_features_rank_copy[c])
            corr, p = stats.spearmanr(original, rank )
            sum_corr = sum_corr + corr
            plt.text(i-1+0.45, 8.1, 'c:'+str(round(corr,1)),fontsize=8)
    plt.title('Correlations sum (c): '+str(round(sum_corr,1)),fontsize=9)
    
    df_transformed = retornRankPositions(df_features_rank_copy)
    df_transformed.to_csv('.'+bar+'output'+bar+'csv'+bar+'df_explanation_analysis_transform_'+model+'_'+test+'.csv',sep=',')

    for col_name in df_transformed.columns:
        df_transformed = df_transformed.rename(columns={col_name: 
                                                        col_name.replace("_x_test",":")
                                                        .replace("_per","per")
                                                        .replace("_","")
                                                        .replace("original","0%")
                                                        .replace("permute","")
                                                        .replace("shap","")
                                                        .replace("skater","")
                                                        .replace("dalex","")
                                                        .replace("lofo","")
                                                        .replace("eli5","")
                                                        .replace("eXirt","")
                                                        })


    df_transposed = df_transformed.transpose()
    bumpchart( 
                df_transposed.copy(), 
                rank_axis_distance=1.05,
                show_rank_axis = False, 
                scatter = True, 
                holes = False,
                line_args = {"linewidth": 3, "alpha": 0.7},
                scatter_args = {"s": 10, "alpha": 0.8},
                color_dic=color_dic
            ) ## bump chart class with nice examples can be found on github

    
    plt.savefig('.'+bar+'output'+bar+'fig'+bar+'ranks_comparations_'+model+'_'+test+'.png')


df = pd.read_csv('.'+bar+'output'+bar+'csv'+bar+'df_explanation_analysis.csv',sep=',',index_col=0)


color_dic = {'mass':'#ff7f0e', 'plas':'#ec96aa', 'pedi':'#aec7e8', 'preg':'#ffbb78', 'age':'#2ca02c', 'skin':'#98df8a', 'insu':'#ff9896', 'pres':'#9467bd'}


model = 'mlp'
test = 'exirt'
df_tmp = df[['eXirt_'+model+'_x_test_original', 'eXirt_'+model+'_x_test_4%_permute', 'eXirt_'+model+'_x_test_6%_permute', 'eXirt_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)


model = 'lgbm'
test = 'exirt'
df_tmp = df[['eXirt_'+model+'_x_test_original', 'eXirt_'+model+'_x_test_4%_permute', 'eXirt_'+model+'_x_test_6%_permute', 'eXirt_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'knn'
test = 'exirt'
df_tmp = df[['eXirt_'+model+'_x_test_original', 'eXirt_'+model+'_x_test_4%_permute', 'eXirt_'+model+'_x_test_6%_permute', 'eXirt_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'dt'
test = 'exirt'
df_tmp = df[['eXirt_'+model+'_x_test_original', 'eXirt_'+model+'_x_test_4%_permute', 'eXirt_'+model+'_x_test_6%_permute', 'eXirt_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)




model = 'mlp'
test = 'shap'
df_tmp = df[['shap_'+model+'_x_test_original', 'shap_'+model+'_x_test_4%_permute', 'shap_'+model+'_x_test_6%_permute', 'shap_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'lgbm'
test = 'shap'
df_tmp = df[['shap_'+model+'_x_test_original', 'shap_'+model+'_x_test_4%_permute', 'shap_'+model+'_x_test_6%_permute', 'shap_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'knn'
test = 'shap'
df_tmp = df[['shap_'+model+'_x_test_original', 'shap_'+model+'_x_test_4%_permute', 'shap_'+model+'_x_test_6%_permute', 'shap_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'dt'
test = 'shap'
df_tmp = df[['shap_'+model+'_x_test_original', 'shap_'+model+'_x_test_4%_permute', 'shap_'+model+'_x_test_6%_permute', 'shap_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)




model = 'mlp'
test = 'eli5'
df_tmp = df[['eli5_'+model+'_x_test_original', 'eli5_'+model+'_x_test_4%_permute', 'eli5_'+model+'_x_test_6%_permute', 'eli5_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'lgbm'
test = 'eli5'
df_tmp = df[['eli5_'+model+'_x_test_original', 'eli5_'+model+'_x_test_4%_permute', 'eli5_'+model+'_x_test_6%_permute', 'eli5_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'knn'
test = 'eli5'
df_tmp = df[['eli5_'+model+'_x_test_original', 'eli5_'+model+'_x_test_4%_permute', 'eli5_'+model+'_x_test_6%_permute', 'eli5_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'dt'
test = 'eli5'
df_tmp = df[['eli5_'+model+'_x_test_original', 'eli5_'+model+'_x_test_4%_permute', 'eli5_'+model+'_x_test_6%_permute', 'eli5_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)





model = 'mlp'
test = 'dalex'
df_tmp = df[['dalex_'+model+'_x_test_original', 'dalex_'+model+'_x_test_4%_permute', 'dalex_'+model+'_x_test_6%_permute', 'dalex_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'lgbm'
test = 'dalex'
df_tmp = df[['dalex_'+model+'_x_test_original', 'dalex_'+model+'_x_test_4%_permute', 'dalex_'+model+'_x_test_6%_permute', 'dalex_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'knn'
test = 'dalex'
df_tmp = df[['dalex_'+model+'_x_test_original', 'dalex_'+model+'_x_test_4%_permute', 'dalex_'+model+'_x_test_6%_permute', 'dalex_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'dt'
test = 'dalex'
df_tmp = df[['dalex_'+model+'_x_test_original', 'dalex_'+model+'_x_test_4%_permute', 'dalex_'+model+'_x_test_6%_permute', 'dalex_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)




model = 'mlp'
test = 'lofo'
df_tmp = df[['lofo_'+model+'_x_test_original', 'lofo_'+model+'_x_test_4%_permute', 'lofo_'+model+'_x_test_6%_permute', 'lofo_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'lgbm'
test = 'lofo'
df_tmp = df[['lofo_'+model+'_x_test_original', 'lofo_'+model+'_x_test_4%_permute', 'lofo_'+model+'_x_test_6%_permute', 'lofo_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'knn'
test = 'lofo'
df_tmp = df[['lofo_'+model+'_x_test_original', 'lofo_'+model+'_x_test_4%_permute', 'lofo_'+model+'_x_test_6%_permute', 'lofo_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'dt'
test = 'lofo'
df_tmp = df[['lofo_'+model+'_x_test_original', 'lofo_'+model+'_x_test_4%_permute', 'lofo_'+model+'_x_test_6%_permute', 'lofo_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)


model = 'mlp'
test = 'skater'
df_tmp = df[['skater_'+model+'_x_test_original', 'skater_'+model+'_x_test_4%_permute', 'skater_'+model+'_x_test_6%_permute', 'skater_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'lgbm'
test = 'skater'
df_tmp = df[['skater_'+model+'_x_test_original', 'skater_'+model+'_x_test_4%_permute', 'skater_'+model+'_x_test_6%_permute', 'skater_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'knn'
test = 'skater'
df_tmp = df[['skater_'+model+'_x_test_original', 'skater_'+model+'_x_test_4%_permute', 'skater_'+model+'_x_test_6%_permute', 'skater_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)

model = 'dt'
test = 'skater'
df_tmp = df[['skater_'+model+'_x_test_original', 'skater_'+model+'_x_test_4%_permute', 'skater_'+model+'_x_test_6%_permute', 'skater_'+model+'_x_test_10%_permute']]
plotBumpChart(df_tmp,model,test,color_dic)







