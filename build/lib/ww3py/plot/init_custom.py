import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/reg_storm_ctrl/info/plots/Helvetica.ttf')
mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/reg_storm_ctrl/info/plots/Helvetica-Light.ttf')
mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/reg_storm_ctrl/info/plots/Helvetica-Bold.ttf')

newparams = {'axes.grid': False,
             'lines.linewidth': 1,
             'ytick.labelsize':12,
             'xtick.labelsize':12,
             'axes.labelsize':13,
             'axes.titlesize':13,
             'legend.fontsize':14,
             'figure.titlesize':14,
             'font.family':'Helvetica Light'}
plt.rcParams.update(newparams)
