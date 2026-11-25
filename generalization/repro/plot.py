from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

archive=Path(__file__).resolve().parent/'data'
validation=archive/'validation_810'
initial=archive/'initial_90'
summary=pd.read_csv(validation/'condition_summary.csv')
effects=pd.read_csv(validation/'paired_effects.csv')
first=pd.read_csv(initial/'exploratory_interaction.csv')

def em(method, extra):
    s=summary[(summary.method.eq(method)) & (summary.extra_target_tokens.eq(extra))]
    return float(s.em.mean())

fig,axes=plt.subplots(1,2,figsize=(11.2,4.5),gridspec_kw={'width_ratios':[1.1,1]})
fig.patch.set_facecolor('white')
colors={'Baseline':'#586777','LoRA':'#147d78'}
ax=axes[0]
xs=np.array([0,1])
width=.34
for i,method in enumerate(('Baseline','LoRA')):
    vals=[em(method,0),em(method,1024)]
    bars=ax.bar(xs+(-.5 if i==0 else .5)*width, vals, width,
                color=colors[method],label=method)
    for bar,val in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,val+1.2,f'{val:.2f}%',ha='center',va='bottom',fontsize=10)
ax.set_xticks(xs,['Original','+1,024 tokens'])
ax.set_ylim(0,100)
ax.set_ylabel('Exact match (%)')
ax.set_title('810-question validation: EM scores',loc='left',weight='bold')
ax.legend(frameon=False,loc='upper right')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',color='#d9dfe3',linewidth=.7)
ax.set_axisbelow(True)

ax=axes[1]
row90=first[first.metric.eq('em')].iloc[0]
row810=effects[(effects.metric.eq('em')) & (effects.contrast.eq('LoRA_minus_Baseline'))].iloc[0]
items=[('Initial 90',float(row90.difference_in_change_pp),float(row90.ci95_low_pp),float(row90.ci95_high_pp),'#6b7884'),
       ('Remaining 810',float(row810.change_pp),float(row810.ci95_low_pp),float(row810.ci95_high_pp),colors['LoRA'])]
for y,(label,est,low,high,color) in zip([1,0],items):
    ax.plot([low,high],[y,y],color=color,lw=2.4)
    ax.plot([low,low],[y-.08,y+.08],color=color,lw=2)
    ax.plot([high,high],[y-.08,y+.08],color=color,lw=2)
    ax.scatter([est],[y],s=95,color=color,zorder=4)
    ax.text(high+.5,y,f'{est:+.2f}  [{low:+.2f}, {high:+.2f}]',va='center',fontsize=9)
ax.axvline(0,color='#303b45',lw=1)
ax.set_yticks([1,0],['Initial 90','Remaining 810'])
ax.set_ylim(-.55,1.55)
ax.set_xlim(-6,27)
ax.set_xlabel('LoRA change − Baseline change (percentage points)')
ax.set_title('Paired difference in EM change',loc='left',weight='bold')
ax.spines[['top','right','left']].set_visible(False)
ax.tick_params(axis='y',length=0)
ax.grid(axis='x',color='#e1e5e9',linewidth=.7)
ax.set_axisbelow(True)

fig.text(.02,.015,'95% percentile intervals: 2,000 article-cluster bootstrap draws. The 810 questions come from the same 45 articles; two fixed distractor seeds.',fontsize=8.5,color='#4d5b65')
fig.subplots_adjust(left=.075,right=.97,top=.88,bottom=.22,wspace=.39)
out=Path(__file__).resolve().parent/'reproduced/generalization_810_quantitative'
out.parent.mkdir(exist_ok=True)
fig.savefig(out.with_suffix('.png'),dpi=200,facecolor='white')
fig.savefig(out.with_suffix('.svg'),facecolor='white')
plt.close(fig)
print(out.with_suffix('.png'),out.with_suffix('.svg'))
