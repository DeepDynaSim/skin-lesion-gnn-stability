from experiment_core import *
sr=json.loads((R/'sensitivity/sensitivity_results.json').read_text());print(DEFAULT)
for parameter,values in CFG['sensitivity_ranges'].items():
    for value in values:
        rows=[r for r in sr if (r['parameter']==parameter and r['value']==value) or (value==DEFAULT[parameter] and r['parameter']=='default')];print(parameter,value,len(rows));assert len(rows)==3
