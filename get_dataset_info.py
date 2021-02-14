from analyze_report_intensity_density_V3 import *


def analyze(f, diameter):
    dir_out = 'results/Density_Intensity_EESL'
    folder_out = makedir(dir_out, diameter)
    f_melt = melt_df(f, diameter)
    show_boxplot(f_melt, folder_out)
    plt.show()


def get_set_info(g, set_idx):
    return list(g[g.Set == set_idx].ID)


def main():
    sns.set()
    # d = 5
    # f = read_pdf(d)
    # print(f.head())
    g = pd.read_excel(r'results\Density_Intensity_EESL\RadPath_AveragedDensity_EESL_JTK.xlsx', 'sets')
    unsup, extra, sextant = [get_set_info(g, s) for s in range(3)]
    sub_unsup = [id for id in sextant if (id in unsup)]
    sub_extra = [id for id in sextant if (id in extra)]
    print()


if __name__ == '__main__':
    main()
