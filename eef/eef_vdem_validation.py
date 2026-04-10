"""
EEF V-Dem Convergent Validity — Politomorphism Engine
======================================================
Tests convergent validity of FIIM v2.1 by correlating IS_agg
with V-Dem Liberal Democracy Index (LDI) for Romania, Hungary,
Poland 2005-2024.

Author : Prof. Serban Gabriel Florin | ORCID: 0009-0000-2266-3356
Project: Politomorphism Engine | OSF: 10.17605/OSF.IO/HYDNZ
"""

import math
import csv
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

VDEM_LDI = {
    "Romania": {
        2005:0.42,2006:0.43,2007:0.44,2008:0.44,2009:0.44,
        2010:0.43,2011:0.43,2012:0.41,2013:0.42,2014:0.43,
        2015:0.44,2016:0.43,2017:0.40,2018:0.39,2019:0.40,
        2020:0.41,2021:0.41,2022:0.42,2023:0.42,2024:0.41,
    },
    "Hungary": {
        2005:0.57,2006:0.57,2007:0.56,2008:0.56,2009:0.55,
        2010:0.52,2011:0.46,2012:0.40,2013:0.36,2014:0.32,
        2015:0.29,2016:0.27,2017:0.25,2018:0.24,2019:0.22,
        2020:0.21,2021:0.20,2022:0.20,2023:0.19,2024:0.19,
    },
    "Poland": {
        2005:0.64,2006:0.65,2007:0.67,2008:0.67,2009:0.67,
        2010:0.67,2011:0.67,2012:0.68,2013:0.68,2014:0.68,
        2015:0.67,2016:0.60,2017:0.54,2018:0.50,2019:0.47,
        2020:0.44,2021:0.44,2022:0.44,2023:0.46,2024:0.49,
    },
}

FIIM_IS = {
    "Romania": {
        2005:0.563,2006:0.563,2007:0.549,2008:0.549,2009:0.538,
        2010:0.538,2011:0.555,2012:0.564,2013:0.536,2014:0.530,
        2015:0.522,2016:0.522,2017:0.530,2018:0.541,2019:0.547,
        2020:0.545,2021:0.545,2022:0.545,2023:0.536,2024:0.576,
    },
    "Hungary": {
        2005:0.500,2006:0.500,2007:0.500,2008:0.500,2009:0.508,
        2010:0.521,2011:0.604,2012:0.589,2013:0.622,2014:0.672,
        2015:0.655,2016:0.660,2017:0.670,2018:0.674,2019:0.678,
        2020:0.684,2021:0.684,2022:0.684,2023:0.684,2024:0.684,
    },
    "Poland": {
        2005:0.466,2006:0.466,2007:0.437,2008:0.437,2009:0.437,
        2010:0.437,2011:0.437,2012:0.423,2013:0.423,2014:0.423,
        2015:0.514,2016:0.477,2017:0.520,2018:0.546,2019:0.617,
        2020:0.575,2021:0.575,2022:0.575,2023:0.561,2024:0.546,
    },
}

YEARS = list(range(2005,2025))
COUNTRIES = ["Romania","Hungary","Poland"]

def mean(x): return sum(x)/len(x)

def pearson_r(x, y):
    mx,my = mean(x),mean(y)
    num = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y))
    den = (sum((xi-mx)**2 for xi in x)*sum((yi-my)**2 for yi in y))**0.5
    return num/den if den!=0 else 0.0

def spearman_rho(x, y):
    def rank(lst):
        s=sorted(enumerate(lst),key=lambda t:t[1])
        r=[0]*len(lst)
        for rv,(oi,_) in enumerate(s,1): r[oi]=rv
        return r
    return pearson_r(rank(x),rank(y))

def t_stat(r,n):
    if abs(r)>=1.0: return float('inf')
    return r*math.sqrt(n-2)/math.sqrt(1-r**2)

def p_approx(t,n):
    df=n-2
    if abs(t)>10: return 0.0001
    z=abs(t)/math.sqrt(1+t**2/df)
    p=2*(1-(0.5*(1+math.erf(z/math.sqrt(2)))))
    return max(p,0.0001)

def interpret_r(r):
    a=abs(r)
    if a>=0.80: return "Very strong"
    elif a>=0.60: return "Strong"
    elif a>=0.40: return "Moderate"
    elif a>=0.20: return "Weak"
    return "Negligible"

def compute():
    results={}
    all_IS,all_LDI=[],[]
    for country in COUNTRIES:
        IS_v=[FIIM_IS[country][y] for y in YEARS]
        LDI_v=[VDEM_LDI[country][y] for y in YEARS]
        r=pearson_r(IS_v,LDI_v)
        rho=spearman_rho(IS_v,LDI_v)
        n=len(IS_v)
        p=p_approx(t_stat(r,n),n)
        results[country]={"pearson_r":round(r,4),"spearman_rho":round(rho,4),
                          "p":round(p,4),"n":n,"IS_vals":IS_v,"LDI_vals":LDI_v,
                          "strength":interpret_r(r)}
        all_IS.extend(IS_v); all_LDI.extend(LDI_v)
    r=pearson_r(all_IS,all_LDI)
    rho=spearman_rho(all_IS,all_LDI)
    n=len(all_IS)
    p=p_approx(t_stat(r,n),n)
    results["POOLED"]={"pearson_r":round(r,4),"spearman_rho":round(rho,4),
                       "p":round(p,4),"n":n,"IS_vals":all_IS,"LDI_vals":all_LDI,
                       "strength":interpret_r(r)}
    return results

def print_results(results):
    print("\n"+"="*70)
    print("  EEF/FIIM — V-Dem Convergent Validity")
    print("  IS_agg vs V-Dem Liberal Democracy Index | 2005-2024")
    print("="*70)
    print(f"\n  {'Entity':<10} {'N':>3}  {'Pearson r':>10} {'p':>7}  {'Spearman r':>11}  Strength")
    print("  "+"─"*65)
    for entity in COUNTRIES+["POOLED"]:
        r=results[entity]
        print(f"  {entity:<10} {r['n']:>3}  {r['pearson_r']:>10.4f} "
              f"{r['p']:>7.4f}  {r['spearman_rho']:>11.4f}  {r['strength']}")

def chart(results, path="eef_chart_vdem_validity.png"):
    if not HAS_MPL: return
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    fig.patch.set_facecolor('#FAFAFA')
    fig.suptitle("FIIM IS_agg vs V-Dem LDI — Convergent Validity",fontsize=13,fontweight='bold')
    colors={"Romania":"#C0392B","Hungary":"#E67E22","Poland":"#2E4A8B"}
    for idx,country in enumerate(COUNTRIES):
        ax=axes[idx]; ax.set_facecolor('#FAFAFA')
        r=results[country]
        ax.scatter(r["IS_vals"],r["LDI_vals"],color=colors[country],alpha=0.7,s=60,zorder=3)
        IS_v,LDI_v=r["IS_vals"],r["LDI_vals"]
        mx,my=mean(IS_v),mean(LDI_v)
        num=sum((xi-mx)*(yi-my) for xi,yi in zip(IS_v,LDI_v))
        den=sum((xi-mx)**2 for xi in IS_v)
        slope=num/den if den!=0 else 0
        xr=[min(IS_v),max(IS_v)]
        ax.plot(xr,[slope*x+my-slope*mx for x in xr],
                color=colors[country],linewidth=2,linestyle='--',alpha=0.8)
        for i,yr in enumerate(YEARS):
            if yr in [2005,2010,2015,2020,2024]:
                ax.annotate(str(yr),(r["IS_vals"][i],r["LDI_vals"][i]),
                            fontsize=7,color='gray',xytext=(4,2),textcoords='offset points')
        ax.set_title(f"{country}\nr={r['pearson_r']:.3f}, p={r['p']:.3f}",
                     fontsize=11,color=colors[country],fontweight='bold')
        ax.set_xlabel("IS_agg (FIIM)",fontsize=10)
        ax.set_ylabel("V-Dem LDI" if idx==0 else "",fontsize=10)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path,dpi=150,bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

def export_csv(results, path="EEF_VDem_Validity.csv"):
    rows=[]
    for country in COUNTRIES:
        r=results[country]
        for i,year in enumerate(YEARS):
            rows.append({"country":country,"year":year,
                         "IS_agg":round(r["IS_vals"][i],4),
                         "VDem_LDI":round(r["LDI_vals"][i],4),
                         "pearson_r":r["pearson_r"],"spearman_rho":r["spearman_rho"],
                         "p":r["p"],"strength":r["strength"]})
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Saved: {path}")

if __name__=="__main__":
    print("\n  Politomorphism Engine — V-Dem Convergent Validity")
    results=compute()
    print_results(results)
    chart(results)
    export_csv(results)
    print("\n  Done.\n")
