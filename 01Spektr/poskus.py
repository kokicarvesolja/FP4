import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
import array_to_latex as a2l
import csv
import pandas as pd
from astropy.io.votable import parse
from astropy.table import QTable, Table, Column

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}) #font na grafih je LaTexov
rc('text', usetex=True)
#
#def fja (x, A, B, C): 
#    return A + B * x + C * np.sqrt(x)
#
#kot = unp.uarray([45.2, 45.4, 47.1, 44.5, 46.0, 47.2], 6 * [0.1])
#
#dolz = unp.uarray([578, 546, 436, 656, 486, 434], 6 * [0]) * 1e-9
#
#par, cov = curve_fit(fja, unp.nominal_values(dolz), unp.nominal_values(kot), sigma=unp.std_devs(kot), absolute_sigma=True)
#
#A, B, C = par
#
#dA, dB, dC = np.sqrt(np.diag(cov))
#
#print(A, dA, B, dB, C, dC)
#
#fig, ax = plt.subplots()
#
#ax.errorbar(1e9 * unp.nominal_values(dolz)[:3], unp.nominal_values(kot)[:3], yerr=unp.std_devs(kot)[:3], color='tab:red', fmt='s', label="Hg")
#
#ax.errorbar(1e9 * unp.nominal_values(dolz)[3:], unp.nominal_values(kot)[3:], yerr=unp.std_devs(kot)[3:], color='tab:orange', fmt='s', label='$H_2$')
#
#dolz_lin = np.linspace(np.min(unp.nominal_values(dolz)) - 20e-9, np.max(unp.nominal_values(dolz)) +  20e-9)
#
#ax.plot(1e9 * dolz_lin, fja(dolz_lin, A, B, C), color='grey', zorder=-1, label='regresija')
#
##ax.set_title(r'Umeritev kotne skale spektroskopa s Hg in $H_2$')
##ax.set_xlabel(r'$\lambda [nm]$')
##ax.set_ylabel(r'$\phi [\degree]$')
##ax.set_xlim(1e9 * np.min(dolz_lin), 1e9 * np.max(dolz_lin))
##ax.legend()
##fig.tight_layout()
##fig.savefig('kalibracija.pdf')
#
#from numba import vectorize
#
#def makefunc_calc_dolz(A, B, C):
#
#    def calc_dolz(kot):
#        return ((-B - np.sqrt(B**2 - 4*A*(C-kot))) / (2*A))**2
#
#    vcalc_dolz = np.vectorize(calc_dolz)
#
#    return vcalc_dolz
#
#vcalc_dolz = makefunc_calc_dolz(A, B, C)
#
#import imageio.v2 as imageio
#
#spectrum = imageio.imread('./01Spektr/sRGB-approx.png', pilmode='RGB')[0]
#
#def colormap(dolz): 
#    i = np.interp(dolz, [380e-9, 710e-9], [0, len(spectrum)]).astype(int)
#    return spectrum[i - 1]/255
#
## varčna žarnica 
#
#kot1 = np.array([47.3, 46.4, 45.8, 45.5, 45.2])
#
#dolz = vcalc_dolz(kot1)
#
#print(dolz)
#
#fig, ax = plt.subplots()
#
#ax.bar(1e9 * dolz, 1, width=2, color=colormap(dolz))
#
#fig.tight_layout()
#fig.savefig('Vzarnica.pdf')
#
#

data = np.linspace(1, 5, 5)

for d in data: 
    print(d)
