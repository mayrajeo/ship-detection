from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

colors5 = ["#005845", "#84C497", "#F28E77", "#64C1CB", "#F3A44C"]
syke5 = ListedColormap(colors5)
colors9 = ["#005845", "#84C497", "#F28E77", "#64C1CB", "#F3A44C", 
           "#B34733", "#006085", "#575756", "#BB5B0F"]
syke9 = ListedColormap(colors9)

custom_lines_5 = [Line2D([0], [0], color=syke5(0.0), lw=4),
                  Line2D([0], [0], color=syke5(.25), lw=4),
                  Line2D([0], [0], color=syke5(.50), lw=4),
                  Line2D([0], [0], color=syke5(.75), lw=4),
                  Line2D([0], [0], color=syke5(1.0), lw=4),]

custom_lines_9 = [Line2D([0], [0], color=syke9(0.0), lw=4),
                  Line2D([0], [0], color=syke9(.125), lw=4),
                  Line2D([0], [0], color=syke9(.25), lw=4),
                  Line2D([0], [0], color=syke9(.375), lw=4),
                  Line2D([0], [0], color=syke9(.50), lw=4),
                  Line2D([0], [0], color=syke9(.625), lw=4),
                  Line2D([0], [0], color=syke9(.75), lw=4),
                  Line2D([0], [0], color=syke9(.875), lw=4),
                  Line2D([0], [0], color=syke9(1.0), lw=4),]