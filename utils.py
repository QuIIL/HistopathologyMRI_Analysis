import os
import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_ind, f_oneway, mannwhitneyu, normaltest, kruskal
from scikit_posthocs import posthoc_conover, posthoc_mannwhitney
import scipy.stats as ss
import itertools
import statsmodels.api as sm

# tissue_types = ['EPI', 'ENUC', 'STR', 'LUM']
tissue_types = ['Epithelium', 'Epithelial Nuclei', 'Stroma', 'Lumen']
img_types = ['T2W', 'ADC']

all_gs = [-1, 0, '3+3', '3+4', '4+3', '4+4', '4+5', '5+5']
grades = [(-1, 'MR-'), (0, 'MR+'), ('3+3/3+4', 'G3'), ('4+3/4+4/4+5/5+5', 'G4')]
conditions = [(0, 'Benign'), (1, 'Cancer')]


def norm_ab(x, a=0, b=1):
    return (b - a) * ((x - x.min()) / (x.max() - x.min())) + a


def standardize(x):
    return (x - x.mean()) / x.std()


def replace(df, target, fieldname):
    for cl in target.split('/'):
        df[fieldname] = df[fieldname].replace(cl, target)
    return df


def replace_gg(df):
    df = replace(df, '3+3/3+4', 'GleasonGrade')
    df = replace(df, '4+3/4+4/4+5/5+5', 'GleasonGrade')
    return df


def reject_outliers(data, m=2.):
    """https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list"""
    d = np.abs(data - np.mean(data))
    mdev = np.mean(d)
    s = d / mdev if mdev else 0.
    data[s >= m] = np.NaN
    return data


def extract_measure(df):
    y_names = tissue_types + img_types
    x = np.tile(tissue_types + img_types, (len(df), 1)).T.reshape([-1, ])
    y = df[y_names].values.T
    # y = ((y - np.nanmin(y, axis=1)[..., np.newaxis]) / (np.nanmax(y, axis=1) - np.nanmin(y, axis=1))[
    #     ..., np.newaxis])
    y = (y - np.nanmean(y, axis=1)[..., np.newaxis]) / np.nanstd(y, axis=1)[..., np.newaxis]
    return x, y, y_names


def extract_measure_inv(df, not_GG=False):
    if not not_GG:
        df_c = replace_gg(df)
        df_c = df_c[df_c.GleasonGrade == '4+3/4+4/4+5/5+5']
    else:
        df_c = df[df.Condition == 1]
    y_names = tissue_types + img_types + ['Involvement', ]
    x = np.tile(tissue_types + img_types, (len(df_c), 1)).T.reshape([-1, ])
    y = df_c[y_names].values.T
    # y = ((y - np.nanmin(y, axis=1)[..., np.newaxis]) / (np.nanmax(y, axis=1) - np.nanmin(y, axis=1))[
    #     ..., np.newaxis])
    y = (y - np.nanmean(y, axis=1)[..., np.newaxis]) / np.nanstd(y, axis=1)[..., np.newaxis]
    return x, y, y_names


def norm_df(df):
    for col in tissue_types + img_types:
        df[col] = (df[col] - df[col].mean()) / df[col].std()  # This happens inplace


def melt_df(df, diameter=5, id_vars=('Condition', 'GleasonGrade'), norm=True, for_scatter=False):
    value_name = f'measure_{diameter}mm' if not for_scatter else 'Density'
    if norm:
        for col in tissue_types + img_types:
            df[col] = (df[col] - df[col].mean()) / df[col].std()  # This happens inplace

    if for_scatter:
        id_vars = list(id_vars) + img_types
        value_vars = tissue_types
    else:
        id_vars = list(id_vars)
        value_vars = tissue_types + img_types
        # df = df[id_vars + [tt for tt in tissue_types] + [img_type for img_type in img_types]]
    df = df[id_vars + value_vars]
    df.set_axis(id_vars + value_vars, inplace=True, axis=1)
    df = pd.melt(
        df, id_vars=id_vars, value_vars=value_vars,
        var_name='ImgType' if not for_scatter else 'Tissue Component', value_name=value_name,
    )
    # if norm:
    #     for vv in (tissue_types + img_types):
    #         tmp = np.array(df[df['ImgType'] == vv][value_name])
    #         # tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    #         tmp = (tmp - tmp.mean()) / tmp.std()
    #         df.loc[df['ImgType'] == vv, value_name] = tmp
    #     for tt in value_vars:
    return df


def read_pdf(diameter):
    """

    :param diameter:
    :return:
    """
    return pd.read_excel('RadPath_AveragedDensity_EESL.xlsx', f'{diameter}mm')


def count_regions(f):
    total_regions = 0
    for gs in all_gs:
        print(gs, len(f[f.GleasonGrade == gs]))
        total_regions += len(f[f.GleasonGrade == gs])
    assert total_regions == len(f)


def makedir(dir_out, diameter):
    folder_out = os.path.join(dir_out, f'{diameter}mm')
    if not os.path.exists(folder_out):
        os.makedirs(folder_out)
    return folder_out


def show_boxplot(f_melt, folder_out=None):
    col_name = f_melt.columns[-1]

    sub_g_merged_G_0 = f_melt.copy()
    sub_g_merged_G_0 = replace(sub_g_merged_G_0, '3+3/3+4', 'GleasonGrade')
    sub_g_merged_G_0 = replace(sub_g_merged_G_0, '4+4/4+5/5+5', 'GleasonGrade')
    sub_g_merged_G_1 = f_melt.copy()
    sub_g_merged_G_1 = replace(sub_g_merged_G_1, '3+3/3+4', 'GleasonGrade')
    sub_g_merged_G_1 = replace(sub_g_merged_G_1, '4+3/4+4/4+5/5+5', 'GleasonGrade')
    fig, ax = plt.subplots(3, 1, num=6, )
    fig.set_size_inches(20, 9)

    sns.catplot(
        "ImgType", col_name, "Condition",
        # x, y, hue,
        hue_order=[-1, 0, 1],
        data=f_melt,
        kind="box",
        height=3,
        aspect=2.5,
        palette="muted",
        legend=False,
        ax=ax[0]
    )
    sns.catplot("ImgType", col_name, "GleasonGrade",
                hue_order=[-1, 0, '3+3/3+4', '4+3', '4+4/4+5/5+5'],
                data=sub_g_merged_G_0,
                kind="box",
                height=3,
                aspect=2.5,
                palette="muted",
                legend=False,
                ax=ax[1],
                )
    sns.catplot("ImgType", col_name, "GleasonGrade",
                hue_order=[-1, 0, '3+3/3+4', '4+3/4+4/4+5/5+5'],
                data=sub_g_merged_G_1,
                kind="box",
                height=3,
                aspect=2.5,
                legend=False,
                ax=ax[2],
                )
    [ax[i].legend(loc='center left', bbox_to_anchor=(1, .7), ncol=1) for i in range(3)]
    [ax[i].set_xlabel('') for i in range(2)]

    # for i in range(7, 10):
    #     plt.close(i)

    if folder_out is not None:
        plt.savefig(f'{folder_out}/boxplot.png', dpi=300)
        plt.close()
    else:
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()


def show_dist(f, folder_out):
    fig, ax = plt.subplots(6, 1, num=0, figsize=(24, 16))
    plt.subplots_adjust(0, None, 1, None)
    plt.subplots_adjust(hspace=0.5)
    for (i, c) in enumerate([-1, 0, 1]):
        for i_tt, tt in enumerate(tissue_types):
            sns.distplot(f[f.Condition == c].__getattr__(tt),
                         norm_hist=True, ax=ax[i_tt])
        sns.distplot(f[f.Condition == c].T2, norm_hist=True, ax=ax[len(tissue_types)])
        sns.distplot(f[f.Condition == c].ADC, norm_hist=True, ax=ax[len(tissue_types) + 1])

    for i in range(len(img_types) + len(tissue_types)):
        title = tissue_types[i] if i < len(tissue_types) else img_types[i - len(tissue_types) - 1]
        xlabel = 'Density' if i < len(tissue_types) else 'Intensity'
        ax[i].set_title(title)
        ax[i].set_xlabel(xlabel)
        ax[i].legend(('MR- Benign', 'MR+ Bengin', 'Cancer'))

    plt.savefig(f'{folder_out}/distplot_w_mr-.png', dpi=300)
    plt.close()


def compare_two_groups(f_melt, group1=0, group2=1, test_name='ttest'):
    sub_f_melt = f_melt.set_index('ImgType')
    col_name = f_melt.columns[-1]
    print(col_name)
    for y_name in sub_f_melt.index.unique():
        d0 = sub_f_melt[sub_f_melt.Condition == group1].loc[y_name]
        d1 = sub_f_melt[sub_f_melt.Condition == group2].loc[y_name]
        if test_name == 'ttest':
            stats = ttest_ind(d0[col_name], d1[col_name], equal_var=False)
        else:
            stats = mannwhitneyu(d0[col_name], d1[col_name])
        print(y_name, '%.3f' % stats[0], '%.3f' % stats[1])
    print()


def test_normality(f_melt):
    sub_f_melt = f_melt.set_index('ImgType')
    col_name = f_melt.columns[-1]
    print(col_name)
    for i, y_name in enumerate(sub_f_melt.index.unique()):
        x = sub_f_melt.loc[y_name][col_name]
        _, pval = normaltest(x)
        print(y_name, '%.3f' % pval)


def test_kruskal(f_melt, group_name='Condition'):
    """If normality and other assumptions are violated, one can use a non-parametric Kruskal-Wallis H test
    (one-way non-parametric ANOVA) to test if samples came from the same distribution.
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial/
    """
    sub_f_melt = f_melt.set_index('ImgType')
    f_melt = replace_gg(f_melt) if group_name == 'GleasonGrade' else f_melt
    col_name = f_melt.columns[-1]
    print(col_name)
    for i, y_name in enumerate(sub_f_melt.index.unique()):
        x = sub_f_melt.loc[y_name]
        x = x.reset_index()
        data = [x.loc[c, col_name].values for c in x.groupby(group_name).groups.values()]
        _, pval = kruskal(*data)
        print(y_name, '%.3f' % pval)


def test_kruskal_posthocs(f_melt, folder_out, group_name='Condition'):
    """
    https://scikit-posthocs.readthedocs.io/en/latest/tutorial/
    """
    f_melt = replace_gg(f_melt) if group_name == 'GleasonGrade' else f_melt
    f_melt.Condition[f_melt.Condition == -1] = 0
    sub_f_melt = f_melt.set_index('ImgType')
    col_name = f_melt.columns[-1]
    print(col_name)

    p_adjust = 'bonferroni'
    suffix = f'_{p_adjust}' if p_adjust is not None else '_none'
    for alternative in ['less', 'greater']:
        results = []
        for i, y_name in enumerate(sub_f_melt.index.unique()):
            x = sub_f_melt.loc[y_name]
            x = x.reset_index()
            # result = posthoc_conover(x, val_col=col_name, group_col=group_name, p_adjust='bonferroni')  # 'holm bonferroni'
            # result = posthoc_conover(x, val_col=col_name, group_col=group_name, p_adjust='bonferroni')  # 'holm bonferroni'
            result = posthoc_mannwhitney(x, val_col=col_name, group_col=group_name, use_continuity=True,
                                         alternative=alternative,
                                         p_adjust=p_adjust, sort=True)
            print(y_name, result)
            results.append(result)
        try:
            with pd.ExcelWriter(f'{folder_out}/kruskal_posthocs_{group_name}_mwn.xlsx', mode='a') as writer:
                pd.concat(results).to_excel(writer, sheet_name=alternative + suffix)
        except:
            pd.concat(results).to_excel(f'{folder_out}/kruskal_posthocs_{group_name}_mwn.xlsx',
                                        sheet_name=alternative + suffix)


def corrcoef(m, col_names):
    n = len(col_names)
    p = np.ones((n, n))
    r = np.ones((n, n))
    for i, j in itertools.product(range(n), range(n)):
        if i == j:
            continue
        # print(f'{col_names[i]} vs {col_names[j]}')
        r[i, j], p[i, j] = ss.pearsonr(m[:, i], m[:, j])
        r[j, i], p[j, i] = r[i, j], p[i, j]
    return r, p


def ols(m, col_names):
    n = len(col_names)
    p = np.ones((n, n))
    r = np.ones((n, n))
    # vix_on_gdp = pd.ols(y=df['VIXCLS'], x=df['GDP'], intercept=True)
    # print(df['VIXCLS'].corr(df['GDP']), vix_on_gdp.f_stat['p-value'])
    for i, j in itertools.product(range(n), range(n)):
        if i == j:
            continue
        # print(f'{col_names[i]} vs {col_names[j]}')
        model = sm.OLS(m[:, i], m[:, j], intercept=True).fit()
        r[i, j], p[i, j] = model.rsquared, model.pvalues
        r[j, i], p[j, i] = r[i, j], p[i, j]
        # print(model.summary())
        # print()
    return r, p


def show_slopes(f, folder_out=None, save_fig=False, class_name='none', class_value=1, method='corrcoef',
                with_inv=False, not_GG=True, save_file=False):
    f = replace_gg(f)
    f = f[f[class_name] == class_value] if class_name != 'none' else f
    if with_inv:
        x, y, y_names = extract_measure_inv(f, not_GG=not_GG)
    else:
        x, y, y_names = extract_measure(f)
    yy = pd.DataFrame(y.T, columns=y_names)

    class_value = str(class_value).replace('/', '_')
    method_name = 'corr' if method == 'corrcoef' else method
    suffix = '_w_inv' if with_inv else ''
    filename = f'{method_name}_{class_name}_{class_value}{suffix}'

    if save_fig:
        corr = yy.corr('pearson')
        corr_mat = np.round(corr.values, 4)
        sns.set(font_scale=1)  # 1.5
        fig = plt.figure('Corr')
        fig.set_size_inches(18, 9)
        corr_ax = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')
        plt.tight_layout()
        [corr_ax.text(i + .35, j + .5, corr_mat[i, j], fontsize=40)
         for i in range(corr_mat.shape[0]) for j in range(corr_mat.shape[1])]
        # plt.show()
        # exit()
        plt.savefig(f'{folder_out}/{filename}_heatmap_standardized.png', dpi=300)
        plt.close()
        return None
    else:
        func = eval(method)
        corr, p_values = func(yy.to_numpy(), yy.columns)
        corr = pd.DataFrame(corr, columns=yy.columns, index=yy.columns)
        p_values = pd.DataFrame(p_values, columns=yy.columns, index=yy.columns)
        corr_table = pd.concat([corr, p_values])
        if save_file:
            corr_table.to_excel(f'{folder_out}/{filename}.xlsx')

        return corr_table


def show_mnlogit(f, class_name='GleasonGrade'):
    f['intercept'] = 1.0
    # cols = [f'pred_{tt}_5mm' for tt in tissue_types] + [it + '_5mm' for it in img_types] + ['intercept', class_name]
    cols = tissue_types + img_types
    # cols = [f'pred_{tt}_5mm' for tt in tissue_types]
    # cols = [f'pred_{tt}_5mm' for tt in tissue_types][:1] + ['intercept', class_name]
    sub_f = f[cols + [class_name, ]].copy()
    sub_f[cols] = norm_ab(sub_f[cols], b=100)  # Pandas automatically applies colomn-wise function
    # sub_f[class_name] = sub_f[class_name].replace(-1, 0)  # Merge MR- and MR+ benign
    sub_f = sub_f[sub_f[class_name] != -1]  # Drop MR- Benign
    # print(sub_f.describe())
    # exit()
    # sub_f[class_name] = sub_f[class_name].replace(-1, '-1: MR- Benign')
    # sub_f[class_name] = sub_f[class_name].replace(0, '0: MR+ Benign')
    sub_f[class_name] = sub_f[class_name].replace(0, '0: MR Benign')
    sub_f = replace(sub_f, '3+3/3+4', class_name)
    sub_f = replace(sub_f, '4+3/4+4/4+5/5+5', class_name)
    # sub_f.boxplot(cols[:-2], by=class_name)
    # plt.show(), exit()
    exp_results = {}
    y = sub_f['GleasonGrade']

    X = sm.add_constant(sub_f[cols[-2:]])
    result = sm.MNLogit(y, X).fit()
    print(result.summary())
    exp = np.exp(result.params)
    exp.columns = ['3+3/3+4', '4+3/4+4/4+5/5+5']
    print(exp)
    exit()


def calculate_pvalues(df):
    """ Calculate p-values in Pearson correlation when using Pandas
    from https://stackoverflow.com/questions/25571882/pandas-columns-correlation-with-statistical-significance"""
    df_cols = pd.DataFrame(columns=df.columns)
    p_values = df_cols.transpose().join(df_cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            p_values[r][c] = round(ss.pearsonr(df[r], df[c])[1], 5)
    return p_values


def gather_corr_tables_non_split():
    corr_tables = []
    for d in [7, 5, 3, 1]:
        f = read_pdf(d)
        norm_df(f)
        corr_table = f[tissue_types + img_types].corr('pearson')
        corr_table.drop(index=img_types, columns=tissue_types, inplace=True)
        corr_table.insert(0, 'Diameter', f'{d}mm')

        p_table = calculate_pvalues(f[tissue_types + img_types])
        p_table.drop(index=img_types, columns=tissue_types, inplace=True)
        p_table.rename(columns={'T2W': 'T2W_P', 'ADC': 'ADC_P'}, inplace=True)
        corr_table = pd.concat((corr_table, p_table), axis=1)

        corr_tables.append(corr_table)

    corr_tables = pd.concat(corr_tables, axis=0).reset_index().rename(columns={'index': 'Tissue'})
    return corr_tables


def get_corr_table(f, columns=None, with_inv=False, merge_grades=False):
    f[tissue_types + img_types].corr('pearson')
    if columns is None:
        columns = img_types
    if with_inv:
        columns = 'Involvement'
        if not merge_grades:
            grades_selected = [grade for grade in grades if grade[1] in ['G3', 'G4']]
        else:
            grades_selected = [condition for condition in conditions if condition[1] == 'Cancer']
    else:
        if not merge_grades:
            grades_selected = grades
        else:
            grades_selected = conditions
            f['Condition'].replace(-1, 0, inplace=True)  # Replace G
    columns = [columns, ] if not isinstance(columns, list) else columns

    corr_tables = []
    for (grade, grade_str) in grades_selected:
        if not merge_grades:
            corr_tables.append(show_slopes(f, class_name='GleasonGrade', class_value=grade, with_inv=with_inv))
        else:
            corr_tables.append(show_slopes(f, class_name='Condition', class_value=grade, with_inv=with_inv))
        # Remove rows with T2, ADC & Keep columns with T2, ADC
        corr_tables[-1].drop(index=columns, inplace=True)
        corr_tables[-1].drop(columns=corr_tables[-1].columns.difference(columns), inplace=True)

        # Reshape corr_table as the last half is the p-value table
        n = len(corr_tables[-1]) // 2
        corr_tables[-1] = pd.concat((corr_tables[-1][:n], corr_tables[-1][n:]), axis=1)
        column_names = columns + [column + ' p-value' for column in columns]
        corr_tables[-1].columns = column_names

        # Insert grades and reset index
        corr_tables[-1].insert(0, 'Grade', grade_str)
        corr_tables[-1].reset_index(inplace=True)

    corr_tables = pd.concat(corr_tables, axis=0)
    p_value_melt = corr_tables.melt(value_vars=column_names[len(columns):], value_name='p-value')['p-value']
    corr_tables = corr_tables.melt(value_vars=columns[:len(columns)],
                                   id_vars=corr_tables.columns.difference(column_names),
                                   value_name='corr', var_name='MR Modality')
    return pd.concat((corr_tables, p_value_melt), axis=1)


def gather_corr_tables():
    corr_tables = []
    for d in [7, 5, 3, 1]:
        corr_tables.append(get_corr_table(read_pdf(d)))
        corr_tables[-1].insert(0, 'diameter', f'{d}mm')
    corr_tables = pd.concat(corr_tables, axis=0)
    corr_tables.rename(columns={'index': 'Tissue', 'Grade': 'Group'}, inplace=True)
    return corr_tables


def savitzky_golay(y, window_size=11, order=1, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


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
    # test_kruskal_posthocs(f_melt, folder_out)
    test_kruskal(f_melt, group_name='GleasonGrade')
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


if __name__ == '__main__':
    # d = 1
    for d in [7, 5, 3, 1]:
        f = read_pdf(d)
        analyze(f, d)
