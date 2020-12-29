import numpy as np
import matplotlib.pyplot as plt

def maximize_window():
	mng = plt.get_current_fig_manager()
	mng.resize(*mng.window.maxsize())