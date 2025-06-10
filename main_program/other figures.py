import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

project_root=os.path.dirname(os.getcwd())
#######################################################OSP
file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","OSP.png") 
fig, ax=plt.subplots(figsize=(10, 2))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_xlim(-1, 11)
ax.set_ylim(-0.5, 1)
ax.set_xticks(np.arange(0, 11, 1))
ax.set_yticks([])
for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)
ax.spines['bottom'].set_position('zero')
blue_circles_x=[0, 0.8 , 1.2, 2.3, 3, 3.8, 4, 6.5]
for x in blue_circles_x:
    ax.add_patch(plt.Circle((x, 0), 0.1, color='red', alpha=0.7))
red_squares_x=[6, 7.1, 8.6, 9.15, 10.3]
for x in red_squares_x:
    ax.add_patch(plt.Rectangle((x-0.1, -0.1), 0.2, 0.2, color='green', alpha=0.7))
split_point=5
ax.text(split_point, 0.4, 'Split point', ha='center', va='center')
ax.arrow(split_point, 0.25, 0, -0.2, head_width=0.05, head_length=0.1, fc='black')
plt.tight_layout()
plt.savefig(os.path.join(file_path))


################################################################################splitting dataset
file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Splitting Dataset.png") 
fig, ax=plt.subplots()
ax.set_xlim(0, 16+0.2)
ax.set_ylim(0, 0.22)
ax.axis('off') 

total_width=16
height=0.2
y_position=0.01

rect=patches.Rectangle((0, y_position), total_width, height,linewidth=3, edgecolor='black', facecolor='none')
ax.add_patch(rect)
#plt.show()

div1=total_width * 2/4 
div2=total_width * 3/4 

ax.plot([div1, div1], [y_position, y_position + height], 'k-', linewidth=2)
ax.plot([div2, div2], [y_position, y_position + height], 'k-', linewidth=2)

ax.text(div1/2, y_position + height/2, 'Training \n (50%)', 
        ha='center', va='center', fontsize=12)
ax.text((div1 + div2)/2, y_position + height/2, 'Validation \n (25%)', 
        ha='center', va='center', fontsize=12)
ax.text((div2 + total_width)/2, y_position + height/2, 'Test \n (25%)', 
        ha='center', va='center', fontsize=12)

plt.savefig(os.path.join(file_path))

