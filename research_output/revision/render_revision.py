"""Current canonical document skill renderer, with native Word as PDF backend."""
from pathlib import Path
import os,sys,importlib.util,subprocess
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'python_packages'))
SKILL=Path(r'C:\Users\MSI-\.codex\plugins\cache\openai-primary-runtime\documents\26.1013.11914\skills\documents\render_docx.py')
POPPLER=Path(r'C:\Users\MSI-\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin');os.environ['PATH']=str(POPPLER)+os.pathsep+os.environ.get('PATH','')
spec=importlib.util.spec_from_file_location('canonical_revision_renderer',SKILL);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
def native_convert(input_path,user_profile,convert_tmp_dir,stem,verbose=False):
    pdf=str(Path(convert_tmp_dir)/(stem+'.pdf'));p=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(R.parent/'export_word.ps1'),'-InputDocx',str(Path(input_path).resolve()),'-OutputPdf',pdf],capture_output=True,text=True);print(p.stdout,flush=True)
    if p.returncode:raise RuntimeError(p.stderr)
    return pdf,p.stdout
mod.convert_to_pdf=native_convert
if __name__=='__main__':mod.main()
