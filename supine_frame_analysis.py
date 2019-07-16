import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *;
import xlrd


def draw_3d():
    path_xlsx = r'C:\Users\Jin Siao\Desktop\0611GM\xlsxfiles_withNCC\Ritikass.xlsx'

    # path_pic = r'C:\Users\Jin Siao\Desktop\0611GM\pic\du.png'

    df = pd.read_excel(path_xlsx)
    df_cut = df.loc[(df['NCC'].astype('float') >= 0.65) & (df['length'].astype('float') >= 100)]
    # first time sreen out (roughly)

    xper = df_cut['X%']
    yper = df_cut['Y%']
    ncc = df_cut['NCC']
    # nomarl data

    ncc_max = df_cut['NCC'].max()
    # find the NCC max point

    ax = plt.subplot(111, projection='3d')

    N = len(df)
    M = len(df_cut)
    print("numbers of data:", len(df))
    print("usable numbers of data:", len(df_cut))

    colors = np.random.rand(M)
    i = df_cut.loc[df_cut['NCC'] == ncc_max]
    x = i['X%']
    y = i['Y%']
    n = i['NCC']
    # ax.scatter(xmax,yper,nmax,marker= 'o',c= 'r')
    print(i)
    ax.scatter(x, y, n, marker='p', c='r')  # red
    # screen out the NCC max and draw it in red star

    # plt.scatter(xper,yper)
    # plt.xlabel('x%')
    # plt.ylabel('y%')
    ax.scatter(xper, yper, ncc, marker='.', c='c', alpha=0.05)  # cyan
    # draw normal data

    ax.set_xlabel('x%')
    ax.set_ylabel('y%')
    ax.set_zlabel('NCC')

    ax.set_xlim(0.3, 0.7)
    ax.set_ylim(0.3, 0.7)

    x_ave = 0.542161813
    y_ave = 0.438687938

    # df= df3[df3.str.contains('inf')==False]

    # df_select_3per = df.loc[(df['X%']>= x_ave * 1.2) & (df['X%'] <= x_ave * 1.2) & (df['Y%'] >= y_ave * 1.2) & (df['Y%']<= y_ave *1.2)]  # 筛

    # *********************************************************************************************************************************************
    df_select_1per = df_cut.loc[
        (df_cut['X%'].astype('float') >= x_ave * 0.99) & (df_cut['X%'].astype('float') <= x_ave * 1.01) & (
                df_cut['Y%'].astype('float') >= y_ave * 0.91) & (df_cut['Y%'].astype('float') <= y_ave * 1.01)]  # 筛选
    # print(df_select_1per)
    len_1per = len(df_select_1per)
    print("similar(+-1%) numbers of data:", len_1per)

    x_similar_1per = df_select_1per['X%']
    y_similar_1per = df_select_1per['Y%']
    n_similar_1per = df_select_1per['NCC']
    ax.scatter(x_similar_1per, y_similar_1per, n_similar_1per, marker='^', c='m')

    # *********************************************************************************************************************************************

    df_select_3per = df_cut.loc[
        (df_cut['X%'].astype('float') >= x_ave * 0.97) & (df_cut['X%'].astype('float') <= x_ave * 1.03) & (
                df_cut['Y%'].astype('float') >= y_ave * 0.97) & (df_cut['Y%'].astype('float') <= y_ave * 1.03)]  # 筛选
    # print(df_select_3per)
    len_3per = len(df_select_3per)
    print("similar(+-3%) numbers of data:", len_3per)

    x_similar_3per = df_select_3per['X%']
    y_similar_3per = df_select_3per['Y%']
    n_similar_3per = df_select_3per['NCC']
    ax.scatter(x_similar_3per, y_similar_3per, n_similar_3per, marker='^', c='g')  # yellow
    # *********************************************************************************************************************************************

    df_select_5per = df_cut.loc[
        (df_cut['X%'].astype('float') >= x_ave * 0.95) & (df_cut['X%'].astype('float') <= x_ave * 1.05) & (
                df_cut['Y%'].astype('float') >= y_ave * 0.95) & (df_cut['Y%'].astype('float') <= y_ave * 1.05)]  # 筛选
    # print(df_select_5per)
    len_5per = len(df_select_5per)
    print("similar(+-5%) numbers of data:", len_5per)

    x_similar_5per = df_select_5per['X%']
    y_similar_5per = df_select_5per['Y%']
    n_similar_5per = df_select_5per['NCC']
    ax.scatter(x_similar_5per, y_similar_5per, n_similar_5per, marker='^', c='y')  # green
    # *********************************************************************************************************************************************
    ave_ncc_1per = np.mean(n_similar_1per)
    ave_ncc_3per = np.mean(n_similar_3per)
    ave_ncc_5per = np.mean(n_similar_5per)

    print("The NCC max:", ncc_max)
    print("The average of ncc with +-1% floating is:",ave_ncc_1per )
    print("The average of ncc with +-3% floating is:", ave_ncc_3per)
    print("The average of ncc with +-5% floating is:", ave_ncc_5per)

    # similarity = n_similar / ncc_max
    # print(similarity)
    # for s in similarity:
    # ave_similarity = np.mean(similarity)
    # print("The NCC max:", ncc_max)

    plt.show()

    return len_1per, len_3per, len_5per
    # plt.savefig(path_pic,dpi=900)


def draw_bar(len_1per, len_3per, len_5per):
    bar_num = 3

    values = [len_1per, len_3per, len_5per]
    index = np.arange(bar_num)
    width = 0.35

    # plt.xlabel('floating')
    #
    # plt.ylabel('numbers')
    #
    # plt.title('numbers of ')

    plt.figure()
    p2 = plt.bar(index, values, width, label="number of frame", color="#87CEFA")
    plt.xticks(index, ('±1%', '±3%', '±5%'))
    for a, b in zip(index, values):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=7)

    plt.show()


def main():
    len_1per, len_3per, len_5per = draw_3d()
    draw_bar(len_1per, len_3per, len_5per)


if __name__ == '__main__':
    main()
