"""A separate completed numerical replay check, before the longer state audit finishes."""
from replication_driver import HERE,RR,EXPECTED,compare
import json
errors=[]
for name in ['summary.json','corruption_summary.json','transfer_reference.json','finite_theory.json','augmentation_verification.json']:
    errors.extend(compare(json.loads((EXPECTED/name).read_text()),json.loads((RR/'analysis'/name).read_text()),name))
errors.extend(compare(json.loads((EXPECTED/'ph2_audit.json').read_text()),json.loads((RR/'data/ph2_audit.json').read_text()),'PH2_processing'))
report={'status':'numeric_replay_passed','regenerated_numeric_entries_compared':len(errors),'maximum_regenerated_numeric_error':max(errors),'scope':'Numerical replay only; the live full saved-state and representative-refit audit is separate'}
(HERE/'analysis/clean_numeric_replay.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
