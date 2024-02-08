import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/Helvetica.ttf')
mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/Helvetica-Light.ttf')
mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/Helvetica-Bold.ttf')

newparams = {'axes.grid': False,
             'lines.linewidth': 1,
             'ytick.labelsize':14,
             'xtick.labelsize':14,
             'axes.labelsize':14,
             'axes.titlesize':14,
             'legend.fontsize':13,
             'figure.titlesize':14,
             'font.family':'Helvetica Light'}
plt.rcParams.update(newparams)
