from utils_analysis import *

sns.set_style('whitegrid')
font_title = {'color': 'black',
              'weight': 'bold',
              'size': 18}
font_axis = {'color': 'black',
             'weight': 'bold',
             'size': 16}
font_tick_size = 15
legend_properties = {'weight': 'normal', 'size': 14}
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 2.5})


def set_legend(fig, ax):
    # assign legened to figureusing bbox coordinates
    x_shift = 0.98
    y_shift = 0.45
    bbox = (
        fig.subplotpars.left + x_shift, fig.subplotpars.top - y_shift, fig.subplotpars.right - fig.subplotpars.left, .1)
    ax.legend(bbox_to_anchor=bbox, loc='lower left', ncol=1, borderaxespad=0., bbox_transform=fig.transFigure,
              frameon=False, title=None, prop=legend_properties)
    # set font weight and font size of the legend title
    ax.get_legend().get_title().set_fontweight('bold')
    ax.get_legend().get_title().set_fontsize(22)


def set_ticks(axes, titles):
    axes = axes.flatten()
#     titles = np.tile(titles, len(axes) // len(titles))
    for i, ax in enumerate(axes):
        if i < len(titles):
            ax.set_title(titles[i], fontdict=font_title)
        else:
            ax.set_title('')
        # call the tick labels of the x and y axes
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontsize(font_tick_size)  # set tick font size
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontsize(font_tick_size)
    return ax


def decorate(sns_plot, titles, skip_legend=False):
    ax = set_ticks(sns_plot.axes, titles)
    if not skip_legend:
        set_legend(sns_plot.fig, ax)
    sns_plot.set_xlabels(fontdict=font_axis)
    sns_plot.set_ylabels(fontdict=font_axis)


def analyze(f, diameter):
    dir_out = 'results/Density_Intensity_EESL'
    folder_out = makedir(dir_out, diameter)
    f_melt = melt_df(f, diameter)
    # show_boxplot(f_melt)
    # show_boxplot(f_melt, folder_out)
    # show_dist(f, folder_out)
    # compare_two_groups(f_melt, test_name='')
    # test_normality(f_melt)
    # test_kruskal(f_melt)
    test_kruskal_posthocs(f_melt, folder_out)
    # test_kruskal(f_melt, group_name='GleasonGrade')
    # test_kruskal_posthocs(f_melt, folder_out, group_name='GleasonGrade')
    return

    # show_slopes(f, folder_out, save_fig=True)
    save_fig = False
    show_slopes(f, folder_out, save_fig=save_fig, class_name='Condition', class_value=-1)
    show_slopes(f, folder_out, save_fig=save_fig, class_name='Condition', class_value=0)
    # show_slopes(f, folder_out, save_fig=save_fig, class_name='Condition', class_value=1)
    show_slopes(f, folder_out, save_fig=save_fig, class_name='GleasonGrade', class_value='3+3/3+4')
    show_slopes(f, folder_out, save_fig=save_fig, class_name='GleasonGrade', class_value='4+3/4+4/4+5/5+5')

    # show_slopes(f, folder_out, save_fig=False, method='ols')
    # show_slopes(f, folder_out, save_fig=False, class_name='Condition', class_value=1, method='ols')
    # show_slopes(f, folder_out, save_fig=False, class_name='GleasonGrade', class_value='3+3/3+4', method='ols')
    # show_slopes(f, folder_out, save_fig=False, class_name='GleasonGrade', class_value='4+3/4+4/4+5/5+5', method='ols')

    # show_slopes(f, folder_out, save_fig=False, class_name='Condition', class_value=1, with_inv=True, not_GG=True)
    # show_slopes(f, folder_out, save_fig=False, class_name='GleasonGrade', class_value='3+3/3+4',
    #             with_inv=True, not_GG=True)
    # show_slopes(f, folder_out, save_fig=False, class_name='GleasonGrade', class_value='4+3/4+4/4+5/5+5',
    #             with_inv=True, not_GG=True)

    # show_mnlogit(f, class_name='GleasonGrade')
    plt.show()


def analyze2(f, diameter):
    f['Condition'].replace(-1, 0, inplace=True)  # Replace G
    norm_df(f)
    corr_tables = []
    corr_table = f[tissue_types + img_types].corr('spearman').corr_table.drop(index=img_types, columns=tissue_types)
    corr_table.inseart(0, 'diameter', f'{d}mm', inplace=True)
    corr_tables.append(corr_table)
    print()
    # corr_table = sns.heatmap(, vmin=-1, vmax=1, annot=True)

    # f_melt = melt_df(f, diameter, for_scatter=True)

    # corr_tables = get_corr_table(f, merge_grades=True)
    # for col in tissue_types + img_types:
    #     f[col] = (f[col] - f[col].mean()) / f[col].std()
    corr_tables = get_corr_table(f, merge_grades=True)
    corr_tables['p-value'] = np.round(corr_tables['p-value'], 3)
    corr_tables['corr'] = np.round(corr_tables['corr'], 4)

    print()

    f_melt = pd.melt(
        f_melt, id_vars=f_melt.columns.difference(img_types), value_vars=img_types,
        var_name='Modality', value_name='Intensity',
    )
    lmplot = sns.lmplot(x='Intensity', y='Density', hue='Condition', data=f_melt, col='Tissue Component', legend=False,
                        row='Modality',
                        aspect=.85).set(xlim=(-5, 5), ylim=(-5, 5))
    decorate(lmplot, f_melt['Tissue Component'].unique(), skip_legend=True)
    lmplot.set_xlabels('Intensity').fig.subplots_adjust(wspace=.02)
    plt.show()
    # with pd.ExcelWriter('results/Density_Intensity_EESL/corr.xlsx', mode='a') as writer:
    #     corr_tables.to_excel(writer, sheet_name=f'{diameter}mm')


def analyze_corr_inv(f, diameter):
    # f_melt = melt_df(f, diameter)
    corr_tables = get_corr_table(f, with_inv=True, merge_grades=True)
    with pd.ExcelWriter('results/Density_Intensity_EESL/corr_inv.xlsx', mode='a') as writer:
        corr_tables.to_excel(writer, sheet_name=f'{diameter}mm')


def main():
    sns.set_style('whitegrid')
    gather_corr_tables_non_split()

    # for d in [7]:
    for d in [7, 5, 3, 1]:
        f = read_pdf(d)
        analyze(f, d)
    exit()
    corr_tables = gather_corr_tables()
    sns.catplot(x='diameter', y='corr', hue='Tissue', row='MR Modality', col='Group', kind="point", data=corr_tables)

    plt.show()
    print()


def make_learning_curves():
    dir_in = r'C:\Users\Minh\Google Drive\Writing\RadPath\Analysis\model_logs'
    from glob import glob
    df = []
    files = glob(dir_in + '/*.csv')
    for file in files:
        df.append(pd.read_csv(file))
        model = 'Semi-supervised Learning' if '_TSA' in file else 'Supervised Learning'
        metric = 'L1 Loss' if 'l1' in file else 'SSIM'
        tissue_type = file[-7:-4]
        tissue_type = 'ENUC' if tissue_type == 'NUC' else tissue_type
        df[-1]['model'] = model
        df[-1]['tissue_type'] = tissue_type
        df[-1]['metric'] = metric
        df[-1].Value = savitzky_golay(df[-1].Value.values, 11, 1)
    df = pd.concat(df, axis=0, ignore_index=True)
    sns.relplot(x='Step', y='Value', hue='model', col='tissue_type', row='metric', data=df, kind='line')
    print('hello')
    plt.show()


if __name__ == '__main__':
    # make_learning_curves()
    main()
