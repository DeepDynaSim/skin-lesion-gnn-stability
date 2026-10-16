from pathlib import Path
import sys,re,json
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'python_packages'))
from lxml import etree
from latex2mathml.converter import convert
transform=etree.XSLT(etree.parse(r'C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL'))
text=(R/'manuscript_template.md.in').read_text(encoding='utf-8');equations=re.findall(r'\$\$(.*?)\$\$',text,re.S);assert len(equations)>=25
records=[]
for i,tex in enumerate(equations,1):
    mathml=etree.fromstring(convert(tex).encode());omml=transform(mathml).getroot();assert omml.tag.endswith('oMath');assert omml.xpath('.//m:t/text()',namespaces={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math'});records.append({'number':i,'latex':tex,'native_OMML':True})
(R/'analysis/equation_structure.json').write_text(json.dumps(records,indent=2));print('Native math transforms verified:',len(records))
